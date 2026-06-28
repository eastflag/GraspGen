# scripts/run_ur10_grasp_simulation.py
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가하여 grasp_gen 패키지 로딩 가능하도록 설정
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import yaml
import trimesh.transformations as tra

# Isaac Sim APIs                                                                                                                                                              
from omni.isaac.kit import SimulationApp

# 1. 시뮬레이션 앱 인스턴스 실행 (GUI 포함)                                                                                                                                   
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import LulaKinematicsSolver

# 2. 시뮬레이션 월드 및 물리 환경 설정                                                                                                                                        
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 3. 물체와 결합된 UR10 로봇 추가                                                                                                                                             
box = world.scene.add(
    DynamicCuboid(
        prim_path="/World/box",
        name="box",
        position=np.array([0.6, 0.0, 0.03]),
        size=0.06,
        color=np.array([0.1, 0.8, 0.1])
        # 녹색
    )
)

# 1단계에서 빌드한 UR10 + Robotiq 2F-85 어셈블리 USD 로드
from omni.isaac.core.utils.stage import add_reference_to_stage
usd_path = "assets/bots/ur10_robotiq_2f_85.usd"
prim_path = "/World/ur10_robotiq"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

robot = world.scene.add(
    Robot(
        prim_path=prim_path,
        name="ur10_robotiq",
        position=np.array([0.0, 0.0, 0.0])
        # 원점 배치
    )
)

# 4. GraspGen 포즈 로딩 및 Z축 오프셋 보정 (2F-85의 짧은 핑거 대응)
yaml_path = "assets/objects/box_grasps.yaml"
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Grasp YAML 파일이 존재하지 않습니다: {yaml_path}")

from grasp_gen.dataset.eval_utils import load_from_isaac_grasp_format
grasps, confidences = load_from_isaac_grasp_format(yaml_path)

# 가장 예측 점수가 높은 첫 번째 Grasp Pose 로드
T_grasp_local = grasps[0]
Z_OFFSET = 0.065  # 140mm 핑거 기준 학습 결과를 85mm 핑거로 보정하기 위한 Z축 전진 오프셋 (6.5cm)
T_grasp_corrected = T_grasp_local @ tra.translation_matrix([0, 0, Z_OFFSET])

# 월드 좌표 상의 물체 포즈를 획득하여 Grasp 위치를 월드 좌표로 변환                                                                                                           
box_pos, box_rot = box.get_world_pose()
T_object_world = tra.quaternion_matrix(box_rot)
T_object_world[:3, 3] = box_pos
T_target_tcp = T_object_world @ T_grasp_corrected

# 5. Inverse Kinematics (IK) 솔버 설정 (UR10)
# Isaac Sim 내장 확장 기능 폴더에서 제공하는 공식 UR10 Lula 설정 파일 경로 매핑
robot_desc_path = "/home/dklee/isaacsim/source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/rmpflow/ur10_robot_description.yaml"
urdf_path = "/home/dklee/isaacsim/source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/ur10_robot.urdf"

kin_solver = LulaKinematicsSolver(
    robot_description_path=robot_desc_path,
    urdf_path=urdf_path
)

target_position = T_target_tcp[:3, 3]
target_orientation = tra.quaternion_from_matrix(T_target_tcp)

# 5. Inverse Kinematics (IK) 솔버 설정 (UR10)
# 팔꿈치가 위를 향하도록 (Elbow-up) 유도하기 위한 warm_start 초기 시드 정의
warm_start = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# IK 계산 수행 (frame_name="ee_link" 및 warm_start 지정)
joint_positions, success = kin_solver.compute_inverse_kinematics(
    frame_name="ee_link",
    target_position=target_position,
    target_orientation=target_orientation,
    warm_start=warm_start
)

# 6. 시뮬레이션 동작 제어 루프
world.reset()

# 시뮬레이션 시작 시 로봇을 흐느적거리지 않는 elbow-up 대기 포즈로 즉각 텔레포트 초기화
initial_joints = np.zeros(robot.num_dof)
initial_joints[:6] = warm_start
robot.set_joint_positions(initial_joints, joint_indices=list(range(robot.num_dof)))

if success:
    print("✓ IK 계산 성공. 로봇 팔이 목표 Grasp Pose로 이동합니다.")

    # 목표 관절각 지정 (1~6번 조인트는 UR10 각도)
    robot.apply_action(
        ArticulationAction(
            joint_positions=joint_positions[:6],
            joint_indices=np.array(list(range(6)))
        )
    )

    # 시뮬레이션 스텝 루프 실행
    gripper_joint_targets = np.array([0.8] * (robot.num_dof - 6))
    for step in range(240):  # 약 4초 시뮬레이션
        # 1. 매 스텝마다 UR10 팔 조인트 제어 명령 유지 (중력에 처지지 않도록 고정)
        robot.apply_action(
            ArticulationAction(
                joint_positions=joint_positions[:6],
                joint_indices=np.array(list(range(6)))
            )
        )

        # 2. 로봇 팔이 도달한 시점(약 1.5초 이후)부터 그리퍼 작동
        if step > 100:
            robot.apply_action(
                ArticulationAction(
                    joint_positions=gripper_joint_targets,
                    joint_indices=np.array(list(range(6, robot.num_dof)))
                )
            )

        world.step(render=True)

    # 움켜쥔 이후 시뮬레이션 상태를 계속 유지하여 창이 유지되도록 함
    print("✓ 동작 시뮬레이션 완료. 창을 유지합니다 (종료하려면 창을 닫거나 터미널에서 Ctrl+C).")
    while simulation_app.is_running():
        # 루프가 유지되는 동안 제어 입력을 계속 유지
        robot.apply_action(
            ArticulationAction(
                joint_positions=joint_positions[:6],
                joint_indices=np.array(list(range(6)))
            )
        )
        robot.apply_action(
            ArticulationAction(
                joint_positions=gripper_joint_targets,
                joint_indices=np.array(list(range(6, robot.num_dof)))
            )
        )
        world.step(render=True)
else:
    print("✗ 해당 Grasp 포즈에 대해 UR10이 도달할 수 있는 관절각(IK 해)을 찾지 못했습니다.")

# 시뮬레이션 종료 처리
simulation_app.close()

