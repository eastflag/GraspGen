# scripts/run_ur10_grasp_simulation.py
import os
import sys
import time

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
grasps, _ = load_from_isaac_grasp_format(yaml_path)

# ── 좌표계 설명 ──────────────────────────────────────────────────────────────
# Isaac Grasp Format의 grasp pose는 그리퍼 base_link(UR10 플랜지에 마운트되는 지점)
# 기준 프레임을 오브젝트 로컬 프레임으로 표현한 변환행렬입니다.
#
# Robotiq 2F-85 구조:
#   base_link (ee_link 마운트 포인트)
#      └─ Z+ 방향으로 0.130m → finger pad 접촉점
#
# GraspGen approach_direction = [0, 0, 1] (Z+이 그리퍼의 접근 방향)
#
# ────────────────────────────────────────────────────────────────────────────
# [Grasp 선택 전략]
# 1순위: 탑다운 grasp (Z+ approach가 세계 아래를 향함)
#   → 그리퍼가 위에서 아래로 내려왔는 직관적인 파지
#   → 접근 방향 Z+(world): Z 컴포넌트 < -0.7
# 2순위 fallback: ee_link Z가 가장 높은 포즈
# ────────────────────────────────────────────────────────────────────────────

GRIPPER_FINGER_DEPTH = 0.130  # 2F-85: base_link → finger pad 접촉점까지 Z+ 거리 (m)
TOPDOWN_APPROACH_THRESHOLD = -0.7  # Z+ 월드 Z 컴포넌트 < 이 값 → 탑다운 파지

T_target_ee = None   # ee_link가 위치해야 할 월드 프레임 변환행렬
chosen_grasp_idx = -1
box_pos_init, box_rot_init = box.get_world_pose()

# Isaac Sim은 quaternion을 [w, x, y, z] 순서로 반환
T_object_world = tra.quaternion_matrix(box_rot_init)  # [w,x,y,z] → 4x4 rotation
T_object_world[:3, 3] = box_pos_init

# ── 1순위: 탑다운 grasp 우선 선택 ─────────────────────────────────────────────────
for idx, grasp_local in enumerate(grasps):
    T_ee_world = T_object_world @ grasp_local
    approach_dir_world = T_ee_world[:3, 2]  # Z+ column = approach direction
    ee_z_world = T_ee_world[2, 3]           # ee_link 세계 z 높이

    # 탑다운 조건:
    #   (1) approach Z+의 세계 Z 컴포넌트가 충분히 음수 (위에서 내려온는 접근)
    #   (2) ee_link 세계 z가 박스 위에 있어야 함
    if (approach_dir_world[2] < TOPDOWN_APPROACH_THRESHOLD and
            ee_z_world > box_pos_init[2] + 0.10):   # ee_link가 박스보다 10cm 위
        T_target_ee = T_ee_world
        chosen_grasp_idx = idx
        break

if T_target_ee is None:
    print("⚠️ 안전한 Grasp 포즈를 찾지 못했습니다. 가장 높은 포즈를 사용합니다.")
    heights = []
    for grasp_local in grasps:
        T_ee_world = T_object_world @ grasp_local
        heights.append(T_ee_world[2, 3])
    chosen_grasp_idx = np.argmax(heights)
    T_target_ee = T_object_world @ grasps[chosen_grasp_idx]

# 디버그: 선택된 grasp 정보 출력
approach_world    = T_target_ee[:3, 2]   # 접근 방향 (Z+ 컴럼)
finger_contact_w  = T_target_ee[:3, 3] + GRIPPER_FINGER_DEPTH * approach_world
print(f"✓ 선택된 Grasp 인덱스: {chosen_grasp_idx}")
print(f"  ee_link 위치 (world): {np.round(T_target_ee[:3, 3], 4)}")
print(f"  접근 방향 Z+  (world): {np.round(approach_world, 3)} ({'top-down' if approach_world[2] < -0.5 else 'side'})")
print(f"  finger pad 접촉  (world): {np.round(finger_contact_w, 4)}")
print(f"  박스 중심             : {np.round(box_pos_init, 4)}")


# 5. Inverse Kinematics (IK) 솔버 설정 (UR10)
# Isaac Sim 내장 확장 기능 폴더에서 제공하는 공식 UR10 Lula 설정 파일 경로 매핑
robot_desc_path = "/home/dklee/isaacsim/source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/rmpflow/ur10_robot_description.yaml"
urdf_path = "/home/dklee/isaacsim/source/extensions/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur10/ur10_robot.urdf"

kin_solver = LulaKinematicsSolver(
    robot_description_path=robot_desc_path,
    urdf_path=urdf_path
)

# IK 타깃 = ee_link가 위치해야 할 월드 프레임 포즈
target_position    = T_target_ee[:3, 3]
target_orientation = tra.quaternion_from_matrix(T_target_ee)  # [w, x, y, z]

# 팔꿈치가 위를 향하도록 (Elbow-up) 유도하기 위한 home 포즈 정의
home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# IK 계산 수행 (frame_name="ee_link": UR10 플랜지 기준)
joint_positions, success = kin_solver.compute_inverse_kinematics(
    frame_name="ee_link",
    target_position=target_position,
    target_orientation=target_orientation,
    warm_start=home_joints
)

print(f"IK 성공 여부: {success}")
if success:
    print(f"  IK ee_link 타깃: {np.round(target_position, 4)}")
    print(f"  IK 해 (관절각):  {np.round(joint_positions[:6], 3)}")

# 6. 시뮬레이션 동작 제어 루프
world.reset()

# 관절 제어 게인(Stiffness, Damping)을 강하게 설정하여 중력 처짐 방지
stiffness = np.array([1e6, 1e6, 1e6, 1e5, 1e5, 1e5, 1e4, 0.0, 0.0, 0.0, 0.0, 0.0])
damping   = np.array([1e4, 1e4, 1e4, 1e3, 1e3, 1e3, 1e2, 0.0, 0.0, 0.0, 0.0, 0.0])
robot.get_articulation_controller().set_gains(kps=stiffness, kds=damping)

# ── [핵심 수정] 로봇을 HOME 포즈로 초기화 ─────────────────────────────────────
# 이전 코드 버그: set_joint_positions로 IK 목표에 텔레포트 → for loop에서 같은
# 각도를 apply해도 이미 목표 위치에 있으므로 이동이 전혀 보이지 않았음.
# 수정: home 포즈에서 시작하여 보간(interpolation)으로 이동 모션을 표현.
initial_joints = np.zeros(robot.num_dof)
initial_joints[:6] = home_joints          # ← 항상 home 포즈에서 시작
robot.set_joint_positions(initial_joints, joint_indices=list(range(robot.num_dof)))

# world.reset() 이후 물리가 안정될 때까지 워밍업 스텝
for _ in range(10):
    robot.apply_action(
        ArticulationAction(
            joint_positions=home_joints,
            joint_indices=np.array(list(range(6)))
        )
    )
    world.step(render=True)

if success:
    grasp_joints = joint_positions[:6]
    print("✓ IK 계산 성공.")
    print(f"  home  → {np.round(home_joints, 3)}")
    print(f"  grasp → {np.round(grasp_joints, 3)}")

    # ── 단계 파라미터 ──────────────────────────────────────────────────────────
    APPROACH_STEPS = 180   # home → grasp 포즈로 이동 (≈ 3초)
    GRASP_STEPS    = 90    # 그리퍼 닫기              (≈ 1.5초)
    LIFT_STEPS     = 90    # 들어올리기               (≈ 1.5초)
    GRIPPER_CLOSE  = 0.8   # 2F-85 닫힘 값 (rad)

    # 들어올릴 목표 포즈: ee_link를 20cm 위로 올리는 IK
    T_lift_ee = T_target_ee.copy()
    T_lift_ee[2, 3] += 0.20
    lift_joints_arr, lift_success = kin_solver.compute_inverse_kinematics(
        frame_name="ee_link",
        target_position=T_lift_ee[:3, 3],
        target_orientation=tra.quaternion_from_matrix(T_lift_ee),
        warm_start=grasp_joints
    )
    lift_joints = lift_joints_arr[:6] if lift_success else grasp_joints
    if lift_success:
        print(f"  lift  → {np.round(lift_joints, 3)}")
    else:
        print("⚠️ 리프트 IK 실패. 그리퍼만 닫은 채 유지합니다.")

    total_steps = APPROACH_STEPS + GRASP_STEPS + LIFT_STEPS

    for step in range(total_steps):

        # ── Phase 1: home → grasp 이동 (smoothstep 보간) ─────────────────────
        if step < APPROACH_STEPS:
            alpha = step / APPROACH_STEPS              # 0.0 → 1.0
            # S-curve (smoothstep) 보간으로 부드러운 가속/감속
            alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)
            interp_joints = (1.0 - alpha_s) * home_joints + alpha_s * grasp_joints
            robot.apply_action(
                ArticulationAction(
                    joint_positions=interp_joints,
                    joint_indices=np.array(list(range(6)))
                )
            )

        # ── Phase 2: 그리퍼 닫기 ─────────────────────────────────────────────
        elif step < APPROACH_STEPS + GRASP_STEPS:
            robot.apply_action(
                ArticulationAction(
                    joint_positions=np.concatenate([grasp_joints, [GRIPPER_CLOSE]]),
                    joint_indices=np.array(list(range(7)))
                )
            )

        # ── Phase 3: 오브젝트를 쥔 채 들어올리기 ─────────────────────────────
        else:
            beta = (step - APPROACH_STEPS - GRASP_STEPS) / LIFT_STEPS  # 0.0 → 1.0
            beta_s = beta * beta * (3.0 - 2.0 * beta)
            interp_lift = (1.0 - beta_s) * grasp_joints + beta_s * lift_joints
            robot.apply_action(
                ArticulationAction(
                    joint_positions=np.concatenate([interp_lift, [GRIPPER_CLOSE]]),
                    joint_indices=np.array(list(range(7)))
                )
            )

        world.step(render=True)
        time.sleep(0.016)  # 60 FPS 실시간 동기화

    # ── 완료 후 창 유지 ───────────────────────────────────────────────────────
    print("✓ 동작 시뮬레이션 완료. 창을 유지합니다 (종료하려면 창을 닫거나 Ctrl+C).")
    final_joints = np.concatenate([lift_joints, [GRIPPER_CLOSE]])
    while simulation_app.is_running():
        robot.apply_action(
            ArticulationAction(
                joint_positions=final_joints,
                joint_indices=np.array(list(range(7)))
            )
        )
        world.step(render=True)
        time.sleep(0.016)  # CPU 과부하 방지

else:
    print("✗ 해당 Grasp 포즈에 대해 UR10이 도달할 수 있는 관절각(IK 해)을 찾지 못했습니다.")
    print(f"  목표 TCP 위치: {np.round(target_position, 4)}")
    print("  → warm_start를 바꾸거나 grasp 포즈를 재선택하세요.")
    while simulation_app.is_running():
        world.step(render=True)
        time.sleep(0.016)

# 시뮬레이션 종료 처리
simulation_app.close()
