import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import argparse
from pathlib import Path
import matplotlib.animation as animation
import cv2
import pickle
from datetime import datetime

class Camera:
    """Class to handle camera parameters and visualization"""
    def __init__(self, K, R, t, name="camera"):
        """
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        """
        self.K = np.array(K)
        self.R = np.array(R)
        self.t = np.array(t).reshape(3, 1)
        self.name = name

        # Camera position in world coordinates
        # If t = -R @ cam_pos, then cam_pos = -R.T @ t
        self.position = (-R.T @ t).flatten()

        # Image dimensions (assuming standard camera setup)
        self.image_width = int(self.K[0, 2] * 2)  # 2 * cx
        self.image_height = int(self.K[1, 2] * 2)  # 2 * cy
        
    def draw_frustum(self, ax, scale=0.5, color='gray', alpha=0.3):
        """Draw camera frustum in 3D space"""
        # Define frustum corners in camera coordinates
        width, height = 0.4 * scale, 0.3 * scale
        depth = 0.6 * scale  # Positive depth points into the scene
        
        corners_cam = np.array([
            [0, 0, 0],  # Camera center
            [-width, -height, depth],  # Bottom-left
            [width, -height, depth],   # Bottom-right
            [width, height, depth],     # Top-right
            [-width, height, depth],    # Top-left
        ])
        
        # Transform to world coordinates
        corners_world = (self.R.T @ corners_cam.T - self.R.T @ self.t).T
        
        # Define frustum faces
        faces = [
            [corners_world[0], corners_world[1], corners_world[2]],  # Bottom
            [corners_world[0], corners_world[2], corners_world[3]],  # Right
            [corners_world[0], corners_world[3], corners_world[4]],  # Top
            [corners_world[0], corners_world[4], corners_world[1]],  # Left
            [corners_world[1], corners_world[2], corners_world[3], corners_world[4]],  # Front
        ]
        
        # Draw frustum
        frustum = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(frustum)
        
        # Draw camera axes (X=red, Y=green, Z=blue)
        axes_length = 0.3 * scale
        axes = np.eye(3) * axes_length
        axes_world = (self.R.T @ axes).T + self.position.flatten()
        
        origin = self.position.flatten()
        # X-axis (red)
        ax.plot([origin[0], axes_world[0, 0]], 
                [origin[1], axes_world[0, 1]], 
                [origin[2], axes_world[0, 2]], 'r-', linewidth=2)
        # Y-axis (green)
        ax.plot([origin[0], axes_world[1, 0]], 
                [origin[1], axes_world[1, 1]], 
                [origin[2], axes_world[1, 2]], 'g-', linewidth=2)
        # Z-axis (blue)
        ax.plot([origin[0], axes_world[2, 0]], 
                [origin[1], axes_world[2, 1]], 
                [origin[2], axes_world[2, 2]], 'b-', linewidth=2)
        
        # Draw viewing direction (camera Z-axis points toward target)
        # R[2,:] is the Z-axis in world coordinates (points toward target)
        view_dir = self.R[2, :] * 0.8 * scale
        ax.plot([origin[0], origin[0] + view_dir[0]],
                [origin[1], origin[1] + view_dir[1]],
                [origin[2], origin[2] + view_dir[2]],
                'y--', linewidth=3, alpha=0.9, label=f'{self.name} view')

    def project_points_3d_to_2d(self, points_3d):
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: Nx3 array of 3D points in world coordinates

        Returns:
            points_2d: Nx2 array of 2D image coordinates
            valid_mask: N boolean array indicating which points are in front of camera
        """
        points_3d = np.array(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)

        # Transform to camera coordinates
        # Using the same convention as multi_cam_self_calib.py: X_cam = R @ X + t
        points_cam = (self.R @ points_3d.T).T + self.t.T

        # Check if points are in front of camera (positive Z)
        valid_mask = points_cam[:, 2] > 0.01  # Small epsilon to avoid division by zero

        # Project to 2D
        points_2d = np.zeros((len(points_3d), 2))
        if np.any(valid_mask):
            valid_points_cam = points_cam[valid_mask]
            # Apply camera intrinsics
            projected = (self.K @ valid_points_cam.T).T
            points_2d[valid_mask] = (projected[:, :2] / projected[:, 2:3])

        return points_2d, valid_mask

    def is_point_in_image(self, point_2d, margin=0):
        """
        Check if 2D point is within image bounds

        Args:
            point_2d: [x, y] coordinates
            margin: pixel margin from image edges

        Returns:
            bool: True if point is within image bounds
        """
        x, y = point_2d
        return (margin <= x <= self.image_width - margin and
                margin <= y <= self.image_height - margin)

class HumanPose3D:
    """Class to handle 3D human pose data and visualization"""
    
    # Define skeleton connections
    SKELETON_CONNECTIONS = [
        # Torso
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 3),
        # Right arm
        (2, 7), (7, 8), (8, 9),
        # Left arm  
        (3, 10), (10, 11), (11, 12),
        # Right leg
        (0, 13), (13, 14), (14, 15),
        # Left leg
        (0, 16), (16, 17), (17, 18),
    ]
    
    def __init__(self, joints_3d):
        """
        joints_3d: Nx3 array of 3D joint positions
        """
        self.joints = np.array(joints_3d)
        self.num_joints = len(self.joints)
        self.bone_lengths = self._calculate_bone_lengths()
        
    def _calculate_bone_lengths(self):
        """Calculate and store bone lengths for consistency"""
        bone_lengths = {}
        for connection in self.SKELETON_CONNECTIONS:
            if connection[0] < self.num_joints and connection[1] < self.num_joints:
                start = self.joints[connection[0]]
                end = self.joints[connection[1]]
                length = np.linalg.norm(end - start)
                bone_lengths[connection] = length
        return bone_lengths
    
    def enforce_bone_lengths(self, target_lengths=None):
        """Enforce consistent bone lengths"""
        if target_lengths is None:
            target_lengths = self.bone_lengths
            
        # Use iterative adjustment to maintain bone lengths
        for _ in range(3):  # Multiple iterations for stability
            for connection in self.SKELETON_CONNECTIONS:
                if connection[0] < self.num_joints and connection[1] < self.num_joints:
                    parent_idx, child_idx = connection
                    parent = self.joints[parent_idx]
                    child = self.joints[child_idx]
                    
                    current_length = np.linalg.norm(child - parent)
                    target_length = target_lengths.get(connection, current_length)
                    
                    if current_length > 1e-6:
                        # Adjust child position to maintain bone length
                        direction = (child - parent) / current_length
                        self.joints[child_idx] = parent + direction * target_length
        
    def draw_skeleton(self, ax, joint_color='orange', linewidth=3, markersize=8):
        """Draw 3D skeleton"""
        # Draw bones
        for connection in self.SKELETON_CONNECTIONS:
            if connection[0] < self.num_joints and connection[1] < self.num_joints:
                start = self.joints[connection[0]]
                end = self.joints[connection[1]]
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       'c-', linewidth=linewidth, alpha=1.0, zorder=10)
        
        # Draw joints
        ax.scatter(self.joints[:, 0], self.joints[:, 1], self.joints[:, 2],
                  c=joint_color, s=markersize**2, 
                  marker='o', edgecolors='black', linewidths=1.5, 
                  alpha=1.0, zorder=11)
    
    def draw_stick(self, ax, hand_indices=[9], stick_length=1.2, two_handed=False, time_param=0.0):
        """Draw a 3D stick attached to hand(s) with dynamic orientation"""
        if len(hand_indices) == 0 or hand_indices[0] >= self.num_joints:
            return
        
        # Get stick endpoints using the dynamic method
        endpoints = self.get_stick_endpoints(hand_indices, stick_length, two_handed,
                                           stick_direction=None, time_param=time_param)
        stick_start, stick_end = endpoints[0], endpoints[1]
        
        # Draw stick
        ax.plot([stick_start[0], stick_end[0]], 
               [stick_start[1], stick_end[1]], 
               [stick_start[2], stick_end[2]], 
               color='purple', linewidth=5, alpha=1.0, zorder=12)
        
        # Draw endpoints
        ax.scatter([stick_start[0]], [stick_start[1]], [stick_start[2]],
                  c='red', s=150, marker='o', 
                  edgecolors='darkred', linewidths=2,
                  alpha=1.0, zorder=13)
        
        ax.scatter([stick_end[0]], [stick_end[1]], [stick_end[2]],
                  c='lime', s=150, marker='o',
                  edgecolors='darkgreen', linewidths=2,
                  alpha=1.0, zorder=13)

    def get_stick_endpoints(self, hand_indices=[9], stick_length=1.2, two_handed=False,
                          stick_direction=None, time_param=0.0, motion_context=None):
        """
        Get the 3D coordinates of stick endpoints with kinematically realistic orientation

        Args:
            hand_indices: List of joint indices for hands holding the stick
            stick_length: Length of the stick
            two_handed: Whether to use two-handed grip
            stick_direction: Custom stick direction vector [3,] (None for automatic)
            time_param: Time parameter for dynamic stick movement [0,1]
            motion_context: Motion context dict with body_velocity, body_orientation, etc.

        Returns:
            stick_endpoints: 2x3 array of [start_point, end_point] in 3D
        """
        if len(hand_indices) == 0 or hand_indices[0] >= self.num_joints:
            return np.zeros((2, 3))

        if two_handed and len(hand_indices) > 1 and hand_indices[1] < self.num_joints:
            # Two-handed grip with enhanced biomechanics
            hand1_pos = self.joints[hand_indices[0]]  # Right hand
            hand2_pos = self.joints[hand_indices[1]]  # Left hand

            # For two-handed grips, use biomechanically plausible grip positions
            hand_separation = np.linalg.norm(hand2_pos - hand1_pos)

            if stick_direction is not None:
                # Use provided direction
                stick_dir = stick_direction / np.linalg.norm(stick_direction)
                grip_center = (hand1_pos + hand2_pos) / 2
            else:
                # Calculate realistic two-handed grip based on shoulder positions
                shoulder1_pos = self.joints[2] if len(self.joints) > 2 else hand1_pos + np.array([-0.2, -0.3, 0.1])
                shoulder2_pos = self.joints[3] if len(self.joints) > 3 else hand2_pos + np.array([0.2, -0.3, 0.1])

                # Stick direction influenced by shoulder line and natural grip
                shoulder_line = shoulder2_pos - shoulder1_pos
                if np.linalg.norm(shoulder_line) > 1e-6:
                    shoulder_line = shoulder_line / np.linalg.norm(shoulder_line)

                # Create motion context if not provided
                if motion_context is None:
                    pelvis_pos = self.joints[0] if len(self.joints) > 0 else (hand1_pos + hand2_pos) / 2 - np.array([0, 0, 0.8])
                    forward_vec = np.array([0, 1, 0])  # Default forward
                    motion_context = {
                        'body_velocity': forward_vec * 0.5,
                        'body_orientation': 0,
                        'pelvis_position': pelvis_pos
                    }

                # For two-handed grips, stick is typically perpendicular to shoulder line
                # but influenced by motion direction and time variation
                body_forward = motion_context.get('body_velocity', np.array([0, 1, 0]))
                body_forward = body_forward / max(np.linalg.norm(body_forward), 1e-6)

                # Base direction: blend of shoulder perpendicular and body forward
                base_stick_dir = np.cross(shoulder_line, np.array([0, 0, 1]))  # Perpendicular to shoulders, horizontal
                if np.linalg.norm(base_stick_dir) < 1e-6:
                    base_stick_dir = body_forward
                else:
                    base_stick_dir = base_stick_dir / np.linalg.norm(base_stick_dir)

                # Add dynamic motion based on time
                t = time_param * 2 * np.pi
                motion_variation = np.array([
                    0.15 * np.sin(t * 1.8),    # Side-to-side variation
                    0.3 * np.cos(t * 1.2),     # Forward-back variation
                    0.1 * np.sin(t * 2.1)      # Vertical variation
                ])

                stick_dir = base_stick_dir * 0.7 + body_forward * 0.2 + motion_variation * 0.1
                stick_dir = stick_dir / np.linalg.norm(stick_dir)

                # Grip center positioned between hands but slightly biased toward dominant hand
                dominant_bias = 0.1  # Slight bias toward hand1 (right hand typically dominant)
                grip_center = hand1_pos * (0.5 + dominant_bias) + hand2_pos * (0.5 - dominant_bias)

            # Two-handed grip: stick extends symmetrically from grip center
            stick_start = grip_center - stick_dir * stick_length * 0.5
            stick_end = grip_center + stick_dir * stick_length * 0.5

        else:
            # Single-handed grip with enhanced kinematics
            hand_pos = self.joints[hand_indices[0]]

            if stick_direction is not None:
                # Use provided direction
                stick_dir = stick_direction / np.linalg.norm(stick_direction)
            else:
                # Enhanced dynamic stick direction based on full body kinematics
                if motion_context is None:
                    # Calculate motion context from pose
                    pelvis_pos = self.joints[0] if len(self.joints) > 0 else hand_pos - np.array([0, 0, 0.8])
                    torso_pos = self.joints[1] if len(self.joints) > 1 else hand_pos - np.array([0, 0, 0.3])

                    # Estimate body velocity from pelvis-torso orientation
                    body_forward_est = torso_pos - pelvis_pos
                    body_forward_est[2] = 0  # Project to horizontal plane
                    if np.linalg.norm(body_forward_est) > 1e-6:
                        body_forward_est = body_forward_est / np.linalg.norm(body_forward_est)
                    else:
                        body_forward_est = np.array([0, 1, 0])

                    motion_context = {
                        'body_velocity': body_forward_est * 0.8,
                        'body_orientation': np.arctan2(body_forward_est[1], body_forward_est[0]),
                        'pelvis_position': pelvis_pos
                    }

                stick_dir = self._calculate_dynamic_stick_direction(hand_pos, time_param, motion_context)

            # Single-handed grip: stick extends primarily forward from hand
            # Grip typically 20% from stick end for control
            stick_start = hand_pos - stick_dir * stick_length * 0.2
            stick_end = hand_pos + stick_dir * stick_length * 0.8

        return np.array([stick_start, stick_end])

    def _calculate_dynamic_stick_direction(self, hand_pos, time_param, motion_context=None):
        """
        Calculate dynamic stick direction based on time, body kinematics, and motion context

        Args:
            hand_pos: Hand position [3,]
            time_param: Time parameter [0,1] for motion variation
            motion_context: Dictionary with motion info (velocity, body_orientation, etc.)

        Returns:
            stick_direction: Normalized direction vector [3,] considering realistic kinematics
        """
        # Get anatomical reference points
        shoulder_pos = self.joints[2] if len(self.joints) > 2 else hand_pos + np.array([-0.2, -0.3, 0.1])
        elbow_pos = self.joints[7] if len(self.joints) > 7 else hand_pos + np.array([-0.1, -0.15, 0])
        torso_pos = self.joints[1] if len(self.joints) > 1 else hand_pos + np.array([0, -0.4, -0.1])
        head_pos = self.joints[5] if len(self.joints) > 5 else hand_pos + np.array([0, -0.3, 0.3])

        # Calculate body kinematics
        shoulder_to_hand = hand_pos - shoulder_pos
        arm_direction = shoulder_to_hand / max(np.linalg.norm(shoulder_to_hand), 1e-6)

        torso_to_head = head_pos - torso_pos
        body_up_dir = torso_to_head / max(np.linalg.norm(torso_to_head), 1e-6)

        # Extract motion context
        if motion_context is None:
            motion_context = {'body_velocity': np.array([0, 0.5, 0]), 'body_orientation': 0}

        body_velocity = motion_context.get('body_velocity', np.array([0, 0.5, 0]))
        body_forward = body_velocity / max(np.linalg.norm(body_velocity), 1e-6)

        # Time-based motion patterns with realistic biomechanics
        t = time_param * 2 * np.pi

        # Pattern 1: Natural arm extension following body motion
        # Stick tends to align with arm extension but biased toward movement direction
        natural_extension = arm_direction * 0.7 + body_forward * 0.3

        # Pattern 2: Sport-specific motions (javelin, pole vault, stick handling)
        sport_weight = 0.5 + 0.3 * np.sin(t * 1.2)  # Varies sport influence

        # Javelin-like motion: stick aligns with shoulder-hand line, pointing forward-up
        javelin_dir = arm_direction + np.array([0, 0, 0.1])  # Slight upward bias

        # Pole/stick handling: more vertical component, following hand motion
        pole_dir = np.array([
            0.1 * np.sin(t * 2),  # Side-to-side variation
            0.7,                  # Forward dominance
            0.4 + 0.2 * np.sin(t * 1.5)  # Variable upward component
        ])

        # Pattern 3: Biomechanically constrained circular motion
        # Limited by shoulder/elbow joint constraints
        max_deviation = 25 * np.pi / 180  # 25-degree max deviation from natural arm line
        circle_amplitude = np.sin(max_deviation) * 0.6

        # Perpendicular to arm for lateral motion
        arm_perp1 = np.cross(arm_direction, body_up_dir)
        arm_perp1 = arm_perp1 / max(np.linalg.norm(arm_perp1), 1e-6)
        arm_perp2 = np.cross(arm_direction, arm_perp1)

        circular_deviation = (circle_amplitude * np.cos(t * 2.5) * arm_perp1 +
                            circle_amplitude * np.sin(t * 2.5) * arm_perp2)
        constrained_circular = arm_direction + circular_deviation

        # Blend patterns with biomechanical realism
        phase = (time_param * 2) % 1  # Two main phases

        if phase < 0.5:
            # Phase 1: Natural extension with sport-specific bias
            weight = phase / 0.5
            base_direction = (1 - weight) * natural_extension + weight * (
                sport_weight * javelin_dir + (1 - sport_weight) * pole_dir)
        else:
            # Phase 2: Constrained dynamic motion
            weight = (phase - 0.5) / 0.5
            base_direction = (1 - weight) * (sport_weight * javelin_dir + (1 - sport_weight) * pole_dir) + \
                           weight * constrained_circular

        # Apply kinematic constraints
        # 1. Shoulder joint limits - stick can't point directly backward
        shoulder_to_hand_proj = np.dot(base_direction, shoulder_to_hand)
        if shoulder_to_hand_proj < -0.3:  # Prevent extreme backward pointing
            correction = -shoulder_to_hand_proj - 0.3
            base_direction = base_direction + correction * shoulder_to_hand / max(np.linalg.norm(shoulder_to_hand), 1e-6)

        # 2. Elbow joint limits - maintain reasonable elbow angle
        hand_to_elbow = elbow_pos - hand_pos
        if np.linalg.norm(hand_to_elbow) > 1e-6:
            elbow_constraint = np.dot(base_direction, hand_to_elbow)
            if elbow_constraint > 0.7:  # Prevent hyper-extension
                correction = elbow_constraint - 0.7
                base_direction = base_direction - correction * hand_to_elbow / np.linalg.norm(hand_to_elbow)

        # 3. Ground collision avoidance - prevent stick from pointing into ground
        if base_direction[2] < -0.5:
            base_direction[2] = max(base_direction[2], -0.3)

        # 4. Natural grip orientation - stick tends to align with forearm
        forearm_dir = hand_pos - elbow_pos
        if np.linalg.norm(forearm_dir) > 1e-6:
            forearm_dir = forearm_dir / np.linalg.norm(forearm_dir)
            forearm_weight = 0.15  # Moderate influence of forearm direction
            base_direction = (1 - forearm_weight) * base_direction + forearm_weight * forearm_dir

        # Final smoothing and normalization
        stick_dir = base_direction / max(np.linalg.norm(base_direction), 1e-6)

        return stick_dir


def set_axes_equal(ax):
    """
    Sets the 3D plot axes to have an equal aspect ratio.

    Args:
        ax: A matplotlib 3D axis object.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def generate_half_sphere_cameras(n_cameras, radius=4, target=[0, 0, 0.9], min_elevation=10):
    """Generate N cameras distributed evenly over a half-sphere using spiral method"""
    cameras = []
    
    # Use spiral distribution for even sampling on hemisphere
    indices = np.arange(0, n_cameras, dtype=float) + 0.5
    
    # Azimuth angles - evenly distributed around circle
    theta = np.arccos(1 - 2 * indices / (2 * n_cameras))  # Better distribution
    phi = np.pi * (1 + 5**0.5) * indices  # Golden angle
    
    for i in range(n_cameras):
        # Limit elevation to hemisphere (above minimum elevation)
        elevation = min(theta[i], np.pi/2 - np.radians(min_elevation))
        azimuth = phi[i]
        
        # Convert spherical to Cartesian
        x = radius * np.sin(elevation) * np.cos(azimuth)
        y = radius * np.sin(elevation) * np.sin(azimuth)
        z = radius * np.cos(elevation) + target[2]
        
        cam_pos = np.array([x, y, z])
        
        # Calculate camera orientation
        # For the calibration code convention where X_cam = R @ X + t and points in front have Z > 0:
        # We need to set up R and t such that target points have positive Z in camera coordinates

        # Direction from camera toward target
        to_target = np.array(target) - cam_pos
        to_target = to_target / np.linalg.norm(to_target)

        # Camera coordinate system:
        # Z-axis: points FROM camera TOWARD target (positive Z = in front of camera)
        z_axis = to_target

        # Create orthogonal coordinate system
        world_up = np.array([0, 0, 1])

        # X-axis points to the right when looking toward target
        x_axis = np.cross(z_axis, world_up)
        if np.linalg.norm(x_axis) < 1e-6:
            # Handle case when z_axis is parallel to world_up
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis completes right-handed coordinate system
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Build rotation matrix: R transforms world coordinates to camera coordinates
        # R = [x_axis, y_axis, z_axis]^T (rows are the camera axes in world coordinates)
        R = np.array([x_axis, y_axis, z_axis])

        # Translation: t = -R @ cam_pos (position of world origin in camera coordinates)
        t = -R @ cam_pos.reshape(3, 1)
        
        K = np.array([[1600, 0, 600],
                     [0, 1600, 600],
                     [0, 0, 1]])
        
        cameras.append(Camera(K, R, t, f"Camera_{i+1}"))
    
    return cameras

class MotionGenerator:
    """Generate realistic human motion sequences with kinematic motion context"""

    def __init__(self, base_pose, num_frames=30):
        self.base_pose = base_pose
        self.num_frames = num_frames
        self.reference_bone_lengths = base_pose.bone_lengths
        self.motion_contexts = []  # Store motion context for each frame
        
    def generate_circular_walk_motion(self, radius=1.5):
        """Generate walking motion along a circular path with kinematic motion context"""
        frames = []
        base_joints = self.base_pose.joints.copy()
        self.motion_contexts = []  # Reset motion contexts

        for frame in range(self.num_frames):
            t = frame / self.num_frames
            angle = t * 2 * np.pi  # Complete circle

            # Calculate position on circle
            circle_x = radius * np.cos(angle)
            circle_y = radius * np.sin(angle)

            # Calculate walking direction (tangent to circle)
            next_angle = angle + 0.1
            direction = np.array([
                np.cos(next_angle) - np.cos(angle),
                np.sin(next_angle) - np.sin(angle),
                0
            ])
            if np.linalg.norm(direction[:2]) > 0:
                direction = direction / np.linalg.norm(direction)

            # Calculate walking speed (varies slightly for realism)
            walking_speed = 1.2 + 0.3 * np.sin(t * 8 * np.pi)  # Speed varies 0.9-1.5 m/s
            body_velocity = direction * walking_speed

            # Calculate body rotation to face walking direction
            body_rotation = np.arctan2(direction[1], direction[0]) + np.pi/2

            joints = base_joints.copy()

            # Walking phase for natural gait
            walk_phase = t * 4 * np.pi  # 2 complete walk cycles

            # Apply circular translation to all joints
            for i in range(len(joints)):
                joints[i][0] += circle_x
                joints[i][1] += circle_y

                # Apply rotation around pelvis
                if i > 0:  # Don't double-rotate pelvis
                    rel_pos = joints[i][:2] - joints[0][:2]
                    rot_matrix = np.array([
                        [np.cos(body_rotation), -np.sin(body_rotation)],
                        [np.sin(body_rotation), np.cos(body_rotation)]
                    ])
                    new_rel_pos = rot_matrix @ rel_pos
                    joints[i][0] = joints[0][0] + new_rel_pos[0]
                    joints[i][1] = joints[0][1] + new_rel_pos[1]

            # Add vertical bob with realistic cadence variation
            cadence_variation = 1.0 + 0.1 * np.sin(t * 6 * np.pi)  # Slight cadence variation
            bob = np.sin(walk_phase * 2 * cadence_variation) * 0.03
            joints[:, 2] += bob

            # Enhanced arm swing with coordination
            right_swing = np.sin(walk_phase * cadence_variation) * 0.15
            left_swing = np.sin(walk_phase * cadence_variation + np.pi) * 0.15

            # Apply arm swing in local coordinates
            swing_dir = np.array([direction[1], -direction[0], 0])  # Perpendicular to walking

            # Arm swing with shoulder-elbow-hand coordination
            joints[7] += right_swing * swing_dir * 0.8  # Elbow movement
            joints[8] += right_swing * swing_dir * 1.1  # Wrist follows elbow
            joints[9] += right_swing * swing_dir * 1.2  # Hand follows wrist

            joints[10] += left_swing * swing_dir * 0.8
            joints[11] += left_swing * swing_dir * 1.1
            joints[12] += left_swing * swing_dir * 1.2

            # Enhanced leg motion with kinematic constraints
            right_leg_phase = np.sin(walk_phase * cadence_variation)
            left_leg_phase = np.sin(walk_phase * cadence_variation + np.pi)

            # Leg motion with hip-knee-ankle coordination
            step_length = 0.2 * walking_speed / 1.2  # Scale with walking speed

            joints[14] += direction * right_leg_phase * step_length + np.array([0, 0, max(0, right_leg_phase * 0.15)])
            joints[15] += direction * right_leg_phase * step_length * 1.3 + np.array([0, 0, max(0, right_leg_phase * 0.08)])

            joints[17] += direction * left_leg_phase * step_length + np.array([0, 0, max(0, left_leg_phase * 0.15)])
            joints[18] += direction * left_leg_phase * step_length * 1.3 + np.array([0, 0, max(0, left_leg_phase * 0.08)])

            # Store motion context for this frame
            motion_context = {
                'body_velocity': body_velocity,
                'body_orientation': body_rotation,
                'pelvis_position': joints[0].copy(),
                'walking_phase': walk_phase % (2 * np.pi),
                'cadence': cadence_variation,
                'motion_type': 'circular_walk',
                'path_curvature': 1.0 / radius,  # Curvature of circular path
                'time_param': t
            }
            self.motion_contexts.append(motion_context)

            pose = HumanPose3D(joints)
            pose.enforce_bone_lengths(self.reference_bone_lengths)
            frames.append(pose)

        return frames
    
    def generate_triangular_walk_motion(self, side_length=2.0):
        """Generate walking motion along a triangular path"""
        frames = []
        base_joints = self.base_pose.joints.copy()
        
        # Define triangle vertices (equilateral triangle centered at origin)
        vertices = np.array([
            [0, side_length * np.sqrt(3)/3, 0],
            [side_length/2, -side_length * np.sqrt(3)/6, 0],
            [-side_length/2, -side_length * np.sqrt(3)/6, 0]
        ])
        
        for frame in range(self.num_frames):
            t = frame / self.num_frames
            
            # Determine which edge we're on
            edge_progress = (t * 3) % 1
            edge_index = int(t * 3) % 3
            
            # Current and next vertex
            start_vertex = vertices[edge_index]
            end_vertex = vertices[(edge_index + 1) % 3]
            
            # Interpolate position along edge
            position = start_vertex + edge_progress * (end_vertex - start_vertex)
            
            # Walking direction
            direction = end_vertex - start_vertex
            direction = direction / np.linalg.norm(direction)
            
            # Body rotation
            body_rotation = np.arctan2(direction[1], direction[0])
            
            joints = base_joints.copy()
            
            # Walking phase
            walk_phase = t * 6 * np.pi  # 3 complete walk cycles
            
            # Apply translation
            for i in range(len(joints)):
                joints[i][0] += position[0]
                joints[i][1] += position[1]
                
                # Apply rotation
                if i > 0:
                    rel_pos = joints[i][:2] - joints[0][:2]
                    rot_matrix = np.array([
                        [np.cos(body_rotation), -np.sin(body_rotation)],
                        [np.sin(body_rotation), np.cos(body_rotation)]
                    ])
                    new_rel_pos = rot_matrix @ rel_pos
                    joints[i][0] = joints[0][0] + new_rel_pos[0]
                    joints[i][1] = joints[0][1] + new_rel_pos[1]
            
            # Vertical bob
            bob = np.sin(walk_phase * 2) * 0.03
            joints[:, 2] += bob
            
            # Sharper turn animation at vertices
            vertex_proximity = min(edge_progress, 1 - edge_progress)
            if vertex_proximity < 0.1:
                turn_factor = (0.1 - vertex_proximity) / 0.1
                # Lean into turn
                joints[1:7, 0] += turn_factor * 0.05 * np.sign(direction[0])
                joints[1:7, 2] -= turn_factor * 0.02  # Slight dip during turn
            
            # Arm swing
            right_swing = np.sin(walk_phase) * 0.15
            left_swing = np.sin(walk_phase + np.pi) * 0.15
            
            swing_dir = np.array([-direction[1], direction[0], 0])
            joints[7] += right_swing * swing_dir
            joints[8] += right_swing * 1.2 * swing_dir
            joints[9] += right_swing * 1.3 * swing_dir
            
            joints[10] += left_swing * swing_dir
            joints[11] += left_swing * 1.2 * swing_dir
            joints[12] += left_swing * 1.3 * swing_dir
            
            # Leg motion
            right_leg_phase = np.sin(walk_phase)
            left_leg_phase = np.sin(walk_phase + np.pi)
            
            joints[14] += direction * right_leg_phase * 0.2 + np.array([0, 0, max(0, right_leg_phase * 0.15)])
            joints[15] += direction * right_leg_phase * 0.3 + np.array([0, 0, max(0, right_leg_phase * 0.1)])
            
            joints[17] += direction * left_leg_phase * 0.2 + np.array([0, 0, max(0, left_leg_phase * 0.15)])
            joints[18] += direction * left_leg_phase * 0.3 + np.array([0, 0, max(0, left_leg_phase * 0.1)])
            
            pose = HumanPose3D(joints)
            pose.enforce_bone_lengths(self.reference_bone_lengths)
            frames.append(pose)
        
        return frames
    
    def generate_square_walk_motion(self, side_length=2.0):
        """Generate walking motion along a square path"""
        frames = []
        base_joints = self.base_pose.joints.copy()
        
        # Define square vertices (centered at origin)
        half = side_length / 2
        vertices = np.array([
            [half, half, 0],
            [-half, half, 0],
            [-half, -half, 0],
            [half, -half, 0]
        ])
        
        for frame in range(self.num_frames):
            t = frame / self.num_frames
            
            # Determine which edge we're on
            edge_progress = (t * 4) % 1
            edge_index = int(t * 4) % 4
            
            # Current and next vertex
            start_vertex = vertices[edge_index]
            end_vertex = vertices[(edge_index + 1) % 4]
            
            # Interpolate position along edge
            position = start_vertex + edge_progress * (end_vertex - start_vertex)
            
            # Walking direction
            direction = end_vertex - start_vertex
            direction = direction / np.linalg.norm(direction)
            
            # Body rotation
            body_rotation = np.arctan2(direction[1], direction[0])
            
            joints = base_joints.copy()
            
            # Walking phase
            walk_phase = t * 8 * np.pi  # 4 complete walk cycles
            
            # Apply translation
            for i in range(len(joints)):
                joints[i][0] += position[0]
                joints[i][1] += position[1]
                
                # Apply rotation
                if i > 0:
                    rel_pos = joints[i][:2] - joints[0][:2]
                    rot_matrix = np.array([
                        [np.cos(body_rotation), -np.sin(body_rotation)],
                        [np.sin(body_rotation), np.cos(body_rotation)]
                    ])
                    new_rel_pos = rot_matrix @ rel_pos
                    joints[i][0] = joints[0][0] + new_rel_pos[0]
                    joints[i][1] = joints[0][1] + new_rel_pos[1]
            
            # Vertical bob
            bob = np.sin(walk_phase * 2) * 0.03
            joints[:, 2] += bob
            
            # 90-degree turn animation at corners
            corner_proximity = min(edge_progress, 1 - edge_progress)
            if corner_proximity < 0.15:
                turn_factor = (0.15 - corner_proximity) / 0.15
                # Sharp turn mechanics
                joints[1:7, 2] -= turn_factor * 0.03  # Lower center of gravity
                # Lean outward from turn
                lean_dir = np.array([start_vertex[0], start_vertex[1], 0])
                if np.linalg.norm(lean_dir[:2]) > 0:
                    lean_dir = lean_dir / np.linalg.norm(lean_dir)
                    joints[1:7, 0] += turn_factor * 0.06 * lean_dir[0]
                    joints[1:7, 1] += turn_factor * 0.06 * lean_dir[1]
            
            # Arm swing (reduced during turns)
            turn_reduction = 1.0 - (turn_factor if corner_proximity < 0.15 else 0) * 0.5
            right_swing = np.sin(walk_phase) * 0.15 * turn_reduction
            left_swing = np.sin(walk_phase + np.pi) * 0.15 * turn_reduction
            
            swing_dir = np.array([-direction[1], direction[0], 0])
            joints[7] += right_swing * swing_dir
            joints[8] += right_swing * 1.2 * swing_dir
            joints[9] += right_swing * 1.3 * swing_dir
            
            joints[10] += left_swing * swing_dir
            joints[11] += left_swing * 1.2 * swing_dir
            joints[12] += left_swing * 1.3 * swing_dir
            
            # Leg motion (shorter steps during turns)
            step_length = 1.0 - (turn_factor if corner_proximity < 0.15 else 0) * 0.4
            right_leg_phase = np.sin(walk_phase)
            left_leg_phase = np.sin(walk_phase + np.pi)
            
            joints[14] += direction * right_leg_phase * 0.2 * step_length + np.array([0, 0, max(0, right_leg_phase * 0.15)])
            joints[15] += direction * right_leg_phase * 0.3 * step_length + np.array([0, 0, max(0, right_leg_phase * 0.1)])
            
            joints[17] += direction * left_leg_phase * 0.2 * step_length + np.array([0, 0, max(0, left_leg_phase * 0.15)])
            joints[18] += direction * left_leg_phase * 0.3 * step_length + np.array([0, 0, max(0, left_leg_phase * 0.1)])
            
            pose = HumanPose3D(joints)
            pose.enforce_bone_lengths(self.reference_bone_lengths)
            frames.append(pose)
        
        return frames
    
    def generate_figure_eight_motion(self, radius=1.2):
        """Generate walking motion along a figure-8 (lemniscate) path"""
        frames = []
        base_joints = self.base_pose.joints.copy()
        
        for frame in range(self.num_frames):
            t = frame / self.num_frames
            param = t * 2 * np.pi
            
            # Lemniscate parametric equations
            scale = radius * 2
            x = scale * np.cos(param) / (1 + np.sin(param)**2)
            y = scale * np.sin(param) * np.cos(param) / (1 + np.sin(param)**2)
            
            # Calculate walking direction (derivative of position)
            dt = 0.01
            next_param = param + dt
            next_x = scale * np.cos(next_param) / (1 + np.sin(next_param)**2)
            next_y = scale * np.sin(next_param) * np.cos(next_param) / (1 + np.sin(next_param)**2)
            
            direction = np.array([next_x - x, next_y - y, 0])
            if np.linalg.norm(direction[:2]) > 0:
                direction = direction / np.linalg.norm(direction)
            
            # Body rotation
            body_rotation = np.arctan2(direction[1], direction[0])
            
            joints = base_joints.copy()
            
            # Walking phase
            walk_phase = t * 6 * np.pi
            
            # Apply translation
            for i in range(len(joints)):
                joints[i][0] += x
                joints[i][1] += y
                
                # Apply rotation
                if i > 0:
                    rel_pos = joints[i][:2] - joints[0][:2]
                    rot_matrix = np.array([
                        [np.cos(body_rotation), -np.sin(body_rotation)],
                        [np.sin(body_rotation), np.cos(body_rotation)]
                    ])
                    new_rel_pos = rot_matrix @ rel_pos
                    joints[i][0] = joints[0][0] + new_rel_pos[0]
                    joints[i][1] = joints[0][1] + new_rel_pos[1]
            
            # Vertical bob with variation at crossover
            bob = np.sin(walk_phase * 2) * 0.03
            # Add extra movement at figure-8 center
            if abs(x) < 0.1 and abs(y) < 0.1:
                bob += 0.02
            joints[:, 2] += bob
            
            # Banking into curves
            curvature = abs(np.sin(param))
            bank_angle = curvature * 0.05
            joints[1:7, 0] += bank_angle * direction[1]
            joints[1:7, 1] -= bank_angle * direction[0]
            
            # Arm swing
            right_swing = np.sin(walk_phase) * 0.15
            left_swing = np.sin(walk_phase + np.pi) * 0.15
            
            swing_dir = np.array([-direction[1], direction[0], 0])
            joints[7] += right_swing * swing_dir
            joints[8] += right_swing * 1.2 * swing_dir
            joints[9] += right_swing * 1.3 * swing_dir
            
            joints[10] += left_swing * swing_dir
            joints[11] += left_swing * 1.2 * swing_dir
            joints[12] += left_swing * 1.3 * swing_dir
            
            # Leg motion
            right_leg_phase = np.sin(walk_phase)
            left_leg_phase = np.sin(walk_phase + np.pi)
            
            joints[14] += direction * right_leg_phase * 0.2 + np.array([0, 0, max(0, right_leg_phase * 0.15)])
            joints[15] += direction * right_leg_phase * 0.3 + np.array([0, 0, max(0, right_leg_phase * 0.1)])
            
            joints[17] += direction * left_leg_phase * 0.2 + np.array([0, 0, max(0, left_leg_phase * 0.15)])
            joints[18] += direction * left_leg_phase * 0.3 + np.array([0, 0, max(0, left_leg_phase * 0.1)])
            
            pose = HumanPose3D(joints)
            pose.enforce_bone_lengths(self.reference_bone_lengths)
            frames.append(pose)
        
        return frames
    
    def generate_zigzag_walk_motion(self, amplitude=1.5, wavelength=2.0):
        """Generate walking motion along a zigzag path"""
        frames = []
        base_joints = self.base_pose.joints.copy()
        
        total_distance = wavelength * 3  # 3 complete zigzags
        
        for frame in range(self.num_frames):
            t = frame / self.num_frames
            
            # Position along zigzag
            forward_pos = t * total_distance - total_distance/2
            lateral_pos = amplitude * np.sin(t * 6 * np.pi)
            
            position = np.array([lateral_pos, forward_pos, 0])
            
            # Calculate direction (derivative)
            dt = 0.01
            t_next = min(1.0, t + dt)
            forward_next = t_next * total_distance - total_distance/2
            lateral_next = amplitude * np.sin(t_next * 6 * np.pi)
            
            direction = np.array([lateral_next - lateral_pos, forward_next - forward_pos, 0])
            if np.linalg.norm(direction[:2]) > 0:
                direction = direction / np.linalg.norm(direction)
            
            # Body rotation
            body_rotation = np.arctan2(direction[1], direction[0])
            
            joints = base_joints.copy()
            
            # Walking phase
            walk_phase = t * 8 * np.pi
            
            # Apply translation
            for i in range(len(joints)):
                joints[i][0] += position[0]
                joints[i][1] += position[1]
                
                # Apply rotation
                if i > 0:
                    rel_pos = joints[i][:2] - joints[0][:2]
                    rot_matrix = np.array([
                        [np.cos(body_rotation), -np.sin(body_rotation)],
                        [np.sin(body_rotation), np.cos(body_rotation)]
                    ])
                    new_rel_pos = rot_matrix @ rel_pos
                    joints[i][0] = joints[0][0] + new_rel_pos[0]
                    joints[i][1] = joints[0][1] + new_rel_pos[1]
            
            # Vertical bob
            bob = np.sin(walk_phase * 2) * 0.03
            joints[:, 2] += bob
            
            # Quick turns at zigzag points
            turn_sharpness = abs(np.cos(t * 6 * np.pi))
            if turn_sharpness > 0.9:
                turn_factor = (turn_sharpness - 0.9) / 0.1
                joints[1:7, 2] -= turn_factor * 0.02
            
            # Arm swing
            right_swing = np.sin(walk_phase) * 0.15
            left_swing = np.sin(walk_phase + np.pi) * 0.15
            
            swing_dir = np.array([-direction[1], direction[0], 0])
            joints[7] += right_swing * swing_dir
            joints[8] += right_swing * 1.2 * swing_dir
            joints[9] += right_swing * 1.3 * swing_dir
            
            joints[10] += left_swing * swing_dir
            joints[11] += left_swing * 1.2 * swing_dir
            joints[12] += left_swing * 1.3 * swing_dir
            
            # Leg motion
            right_leg_phase = np.sin(walk_phase)
            left_leg_phase = np.sin(walk_phase + np.pi)
            
            joints[14] += direction * right_leg_phase * 0.2 + np.array([0, 0, max(0, right_leg_phase * 0.15)])
            joints[15] += direction * right_leg_phase * 0.3 + np.array([0, 0, max(0, right_leg_phase * 0.1)])
            
            joints[17] += direction * left_leg_phase * 0.2 + np.array([0, 0, max(0, left_leg_phase * 0.15)])
            joints[18] += direction * left_leg_phase * 0.3 + np.array([0, 0, max(0, left_leg_phase * 0.1)])
            
            pose = HumanPose3D(joints)
            pose.enforce_bone_lengths(self.reference_bone_lengths)
            frames.append(pose)
        
        return frames
    
def create_checkerboard_floor(size=10, square_size=1, z_level=0):
    """Create checkerboard floor vertices and faces"""
    faces = []
    colors = []
    
    for i in range(-size, size):
        for j in range(-size, size):
            v1 = [i * square_size, j * square_size, z_level]
            v2 = [(i+1) * square_size, j * square_size, z_level]
            v3 = [(i+1) * square_size, (j+1) * square_size, z_level]
            v4 = [i * square_size, (j+1) * square_size, z_level]
            
            if (i + j) % 2 == 0:
                color = [0.95, 0.95, 0.95, 0.6]
            else:
                color = [0.6, 0.6, 0.6, 0.6]
                
            face = [v1, v2, v3, v4]
            faces.append(face)
            colors.append(color)
    
    return faces, colors

def visualize_animation(cameras, pose_frames, output_path=None, stick_hand_indices=[9], 
                        stick_length=1.2, two_handed=False, fps=10):
    """Visualize animated scene with automatic axis adjustment"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds from all data
    all_points = []
    
    # Collect camera positions
    for cam in cameras:
        all_points.append(cam.position.flatten())
    
    # Collect all pose joint positions across all frames
    for pose in pose_frames:
        all_points.extend(pose.joints)
    
    all_points = np.array(all_points)
    
    # Calculate axis limits with some padding
    padding_factor = 0.2
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_padding = x_range * padding_factor
    y_padding = y_range * padding_factor
    z_padding = z_range * padding_factor
    
    x_limits = [x_min - x_padding, x_max + x_padding]
    y_limits = [y_min - y_padding, y_max + y_padding]
    z_limits = [max(0, z_min - z_padding), z_max + z_padding]  # Keep floor at z=0
    
    # Calculate aspect ratio based on actual ranges
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    z_span = z_limits[1] - z_limits[0]
    
    # Normalize to get aspect ratio (relative to smallest dimension)
    min_span = min(x_span, y_span, z_span)
    aspect_ratio = (x_span/min_span, y_span/min_span, z_span/min_span)
    
    # Create floor based on calculated bounds
    floor_size_x = int(np.ceil(max(abs(x_limits[0]), abs(x_limits[1])) / 1))
    floor_size_y = int(np.ceil(max(abs(y_limits[0]), abs(y_limits[1])) / 1))
    floor_size = max(floor_size_x, floor_size_y)
    floor_faces, floor_colors = create_checkerboard_floor(size=floor_size, square_size=1)
    
    def update_frame(frame_num):
        ax.clear()
        
        # Draw floor
        floor_collection = Poly3DCollection(floor_faces, 
                                           facecolors=floor_colors,
                                           edgecolors='gray',
                                           linewidths=0.1,
                                           alpha=0.6,
                                           zorder=1)
        ax.add_collection3d(floor_collection)
        
        # Draw cameras
        for cam in cameras:
            cam.draw_frustum(ax, scale=1.0, color='lightblue', alpha=0.3)
        
        # Draw current pose
        frame_idx = frame_num % len(pose_frames)
        pose = pose_frames[frame_idx]
        time_param = frame_idx / max(1, len(pose_frames) - 1)
        pose.draw_skeleton(ax, joint_color='orange')
        pose.draw_stick(ax, hand_indices=stick_hand_indices,
                       stick_length=stick_length, two_handed=two_handed,
                       time_param=time_param)
        
        # Set axis properties
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_ylabel('Y (m)', fontsize=14)
        ax.set_zlabel('Z (m)', fontsize=14)
        
        # Apply calculated limits and aspect ratio
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.set_box_aspect(aspect=aspect_ratio)
        
        # Set z-ticks based on data range
        if z_span < 2:
            z_tick_interval = 0.5
        elif z_span < 5:
            z_tick_interval = 1.0
        else:
            z_tick_interval = 2.0
        
        z_ticks = np.arange(z_limits[0], z_limits[1] + z_tick_interval/2, z_tick_interval)
        z_ticks = z_ticks[z_ticks >= 0]  # Only show positive z values
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([f'{tick:.1f}' for tick in z_ticks])
        
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.set_title(f'Frame {frame_num + 1}/{len(pose_frames)}', fontsize=14)
    
    if output_path and output_path.endswith('.gif'):
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(pose_frames),
                                     interval=1000/fps, repeat=True)
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved animation to {output_path}")
    else:
        # Interactive mode
        update_frame(0)
        plt.show()
    
    plt.close()

def generate_sample_pose():
    """Generate a sample base pose"""
    joints_3d = np.array([
        [ 0.00,  0.00,  0.00],   # 0  Pelvis
        [ 0.00,  0.05,  0.30],   # 1  Torso
        [ 0.18,  0.10,  0.58],   # 2  Right shoulder
        [-0.18,  0.10,  0.58],   # 3  Left shoulder
        [ 0.00,  0.12,  0.72],   # 4  Neck
        [ 0.00,  0.14,  0.80],   # 5  Head
        [ 0.00,  0.14,  0.88],   # 6  Head top
        [ 0.36,  0.26,  0.52],   # 7  Right elbow
        [ 0.30,  0.50,  0.46],   # 8  Right wrist
        [ 0.25,  0.58,  0.44],   # 9  Right hand
        [-0.36,  0.26,  0.54],   # 10 Left elbow
        [-0.30,  0.50,  0.48],   # 11 Left wrist
        [-0.25,  0.58,  0.46],   # 12 Left hand
        [ 0.13,  0.02, -0.34],   # 13 Right hip
        [ 0.15,  0.04, -0.60],   # 14 Right knee
        [ 0.16,  0.06, -0.88],   # 15 Right ankle
        [-0.13, -0.02, -0.34],   # 16 Left hip
        [-0.15, -0.04, -0.62],   # 17 Left knee
        [-0.16, -0.06, -0.88],   # 18 Left ankle
    ])
    
    # Shift up from ground
    joints_3d[:, 2] += 0.9
    
    return HumanPose3D(joints_3d)

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D human pose with multiple cameras and animation')
    parser.add_argument('--n_cameras', type=int, default=12, help='Number of cameras in half-sphere')
    parser.add_argument('--camera_radius', type=float, default=4.5, help='Distance of cameras from origin')
    parser.add_argument('--motion_type', choices=['walking', 'javelin', 'static', 'circle', 
                                                  'triangle', 'square', 'figure8', 'zigzag'], 
                       default='walking', help='Type of motion to generate')
    parser.add_argument('--num_frames', type=int, default=30, help='Number of animation frames')
    parser.add_argument('--stick_hand', type=str, default='right', 
                       choices=['right', 'left', 'both'], help='Which hand holds the stick')
    parser.add_argument('--stick_length', type=float, default=1.2, help='Length of the stick')
    parser.add_argument('--output', type=str, help='Output path (.png for image, .gif for animation)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for animation')
    parser.add_argument('--path_size', type=float, default=2.0, 
                       help='Size parameter for geometric paths (radius for circle, side length for others)')
    
    args = parser.parse_args()
    
    # Generate cameras in half-sphere configuration
    print(f"Generating {args.n_cameras} cameras in half-sphere configuration...")
    cameras = generate_half_sphere_cameras(
        n_cameras=args.n_cameras,
        radius=args.camera_radius,
        target=[0, 0, 0.9],
        min_elevation=10
    )
    
    # Generate base pose
    base_pose = generate_sample_pose()
    
    # Determine hand indices for stick
    hand_indices = []
    two_handed = False
    if args.stick_hand == 'right':
        hand_indices = [9]
    elif args.stick_hand == 'left':
        hand_indices = [12]
    elif args.stick_hand == 'both':
        hand_indices = [9, 12]
        two_handed = True
    
    # Generate motion frames
    if args.motion_type == 'static':
        pose_frames = [base_pose]
    else:
        print(f"Generating {args.motion_type} motion with {args.num_frames} frames...")
        motion_gen = MotionGenerator(base_pose, args.num_frames)
        
        if args.motion_type == 'circle':
            pose_frames = motion_gen.generate_circular_walk_motion(radius=args.path_size)
        elif args.motion_type == 'triangle':
            pose_frames = motion_gen.generate_triangular_walk_motion(side_length=args.path_size)
        elif args.motion_type == 'square':
            pose_frames = motion_gen.generate_square_walk_motion(side_length=args.path_size)
        elif args.motion_type == 'figure8':
            pose_frames = motion_gen.generate_figure_eight_motion(radius=args.path_size * 0.6)
        elif args.motion_type == 'zigzag':
            pose_frames = motion_gen.generate_zigzag_walk_motion(amplitude=args.path_size * 0.75, 
                                                                 wavelength=args.path_size)
    
    # Visualize
    print("Creating visualization...")
    visualize_animation(
        cameras=cameras,
        pose_frames=pose_frames,
        output_path=args.output,
        stick_hand_indices=hand_indices,
        stick_length=args.stick_length,
        two_handed=two_handed,
        fps=args.fps
    )
    
    print("Done!")

def generate_synthetic_data(num_cameras=4, n_frames=30, stick_length=1.2,
                          motion_type='circle', path_size=2.0, noise_std=0.5,
                          hand_indices=[9], two_handed=False, camera_radius=4.5):
    """
    Generate synthetic data with 3D poses and 2D projections for multiple cameras

    Args:
        num_cameras: Number of cameras to generate
        n_frames: Number of frames in the motion sequence
        stick_length: Length of the stick object
        motion_type: Type of motion ('circle', 'triangle', 'square', 'figure8', 'zigzag')
        path_size: Size parameter for the motion path
        noise_std: Standard deviation of 2D projection noise
        hand_indices: Indices of joints holding the stick
        two_handed: Whether to use two-handed grip
        camera_radius: Distance of cameras from the target

    Returns:
        Dictionary containing all synthetic data
    """
    print(f"Generating synthetic data with {num_cameras} cameras, {n_frames} frames...")

    # Generate cameras in half-sphere configuration
    cameras = generate_half_sphere_cameras(
        n_cameras=num_cameras,
        radius=camera_radius,
        target=[0, 0, 0.9],
        min_elevation=5  # Lower minimum elevation for better coverage
    )

    # Generate base pose
    base_pose = generate_sample_pose()

    # Generate motion frames
    motion_gen = None  # Initialize for scope
    if motion_type == 'static':
        pose_frames = [base_pose] * n_frames
    else:
        motion_gen = MotionGenerator(base_pose, n_frames)

        if motion_type == 'circle':
            pose_frames = motion_gen.generate_circular_walk_motion(radius=path_size)
        elif motion_type == 'triangle':
            pose_frames = motion_gen.generate_triangular_walk_motion(side_length=path_size)
        elif motion_type == 'square':
            pose_frames = motion_gen.generate_square_walk_motion(side_length=path_size)
        elif motion_type == 'figure8':
            pose_frames = motion_gen.generate_figure_eight_motion(radius=path_size * 0.6)
        elif motion_type == 'zigzag':
            pose_frames = motion_gen.generate_zigzag_walk_motion(amplitude=path_size * 0.75,
                                                               wavelength=path_size)
        else:
            pose_frames = motion_gen.generate_circular_walk_motion(radius=path_size)

    # Generate 3D stick trajectories with enhanced kinematics
    stick_3d_gt = np.zeros((n_frames, 2, 3))
    pose_3d_gt = np.zeros((n_frames, len(base_pose.joints), 3))

    for frame_idx, pose in enumerate(pose_frames):
        # Store 3D pose joints
        pose_3d_gt[frame_idx] = pose.joints

        # Get stick endpoints for this pose with motion context and time-based variation
        time_param = frame_idx / max(1, n_frames - 1)  # Normalize to [0, 1]

        # Extract motion context if available from motion generator
        motion_context = None
        if hasattr(pose, 'motion_context'):
            motion_context = pose.motion_context
        elif motion_gen is not None and hasattr(motion_gen, 'motion_contexts') and frame_idx < len(motion_gen.motion_contexts):
            motion_context = motion_gen.motion_contexts[frame_idx]

        stick_endpoints = pose.get_stick_endpoints(hand_indices, stick_length, two_handed,
                                                  stick_direction=None, time_param=time_param,
                                                  motion_context=motion_context)
        stick_3d_gt[frame_idx] = stick_endpoints

    # Generate 2D projections for each camera
    all_observed_points_2d = []  # For stick endpoints (compatible with existing code)
    all_pose_projections_2d = []  # For full body poses
    all_projection_masks = []    # Which points are visible/in bounds

    valid_cameras = []
    for cam_idx, camera in enumerate(cameras):
        stick_obs = np.zeros((n_frames, 2, 2))  # [frames, endpoints, xy]
        pose_obs = np.zeros((n_frames, len(base_pose.joints), 2))  # [frames, joints, xy]
        visibility_mask = np.zeros((n_frames, len(base_pose.joints)), dtype=bool)

        valid_frames = 0
        for frame_idx in range(n_frames):
            # Project stick endpoints
            stick_endpoints_3d = stick_3d_gt[frame_idx]
            stick_2d, stick_valid = camera.project_points_3d_to_2d(stick_endpoints_3d)

            # Check if stick endpoints are within image bounds and in front of camera
            stick_in_bounds = np.array([
                camera.is_point_in_image(stick_2d[0], margin=20) if stick_valid[0] else False,
                camera.is_point_in_image(stick_2d[1], margin=20) if stick_valid[1] else False
            ])

            # Store stick projections for all valid points (no occlusion filtering)
            # Only filter out points that are behind camera or outside image bounds
            for i in range(2):  # For both endpoints
                if stick_valid[i] and stick_in_bounds[i]:
                    # Add noise to valid stick projections
                    stick_obs[frame_idx, i] = stick_2d[i] + np.random.randn(2) * noise_std
                else:
                    # Mark invalid projections as [0, 0]
                    stick_obs[frame_idx, i] = np.array([0.0, 0.0])

            # Count frame as valid if at least one stick endpoint is visible
            if np.any(stick_in_bounds):
                valid_frames += 1

            # Project full pose (always process, no occlusion-based filtering)
            pose_3d = pose_3d_gt[frame_idx]
            pose_2d, pose_valid = camera.project_points_3d_to_2d(pose_3d)

            # Check which joints are within image bounds and in front of camera
            pose_in_bounds = np.array([
                camera.is_point_in_image(pose_2d[j], margin=20) if pose_valid[j] else False
                for j in range(len(pose_3d))
            ])

            # Store pose projections - only filter out-of-bounds, no occlusion
            for j in range(len(pose_3d)):
                if pose_in_bounds[j]:
                    pose_2d[j] += np.random.randn(2) * noise_std
                else:
                    # Set invalid projections to [0, 0] to mark them as invalid
                    pose_2d[j] = np.array([0.0, 0.0])

            pose_obs[frame_idx] = pose_2d
            visibility_mask[frame_idx] = pose_in_bounds

        # Only include cameras that can see the stick in most frames
        visibility_ratio = valid_frames / n_frames if n_frames > 0 else 0
        print(f"  Camera {cam_idx+1}: {valid_frames}/{n_frames} valid frames ({visibility_ratio:.1%})")

        if valid_frames >= n_frames * 0.3:  # At least 30% of frames visible
            all_observed_points_2d.append(stick_obs)
            all_pose_projections_2d.append(pose_obs)
            all_projection_masks.append(visibility_mask)
            valid_cameras.append(camera)

    print(f"Generated {len(valid_cameras)} valid cameras out of {num_cameras}")

    # Build camera parameters arrays
    Rs_gt = [cam.R for cam in valid_cameras]
    ts_gt = [cam.t.flatten() for cam in valid_cameras]
    K = valid_cameras[0].K if valid_cameras else np.eye(3)

    return {
        'num_cameras': len(valid_cameras),
        'n_frames': n_frames,
        'stick_length': stick_length,
        'motion_type': motion_type,
        'K': K,
        'Rs_gt': Rs_gt,
        'ts_gt': ts_gt,
        'cameras': valid_cameras,
        'stick_3d_gt': stick_3d_gt,
        'pose_3d_gt': pose_3d_gt,
        'all_observed_points_2d': all_observed_points_2d,
        'all_pose_projections_2d': all_pose_projections_2d,
        'all_projection_masks': all_projection_masks,
        'hand_indices': hand_indices,
        'two_handed': two_handed,
        'noise_std': noise_std,
        'generation_params': {
            'path_size': path_size,
            'camera_radius': camera_radius,
            'timestamp': datetime.now().isoformat()
        }
    }

def save_synthetic_data(data, filename=None, data_dir='data'):
    """
    Save synthetic data to disk

    Args:
        data: Dictionary containing synthetic data from generate_synthetic_data
        filename: Optional filename (auto-generated if None)
        data_dir: Directory to save data in

    Returns:
        str: Path to saved file
    """
    Path(data_dir).mkdir(exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        motion = data.get('motion_type', 'unknown')
        filename = f"synthetic_data_{motion}_{data['num_cameras']}cams_{data['n_frames']}frames_{timestamp}.pkl"

    filepath = Path(data_dir) / filename

    # Convert cameras to serializable format
    data_to_save = data.copy()
    data_to_save['cameras'] = [
        {
            'K': cam.K.tolist(),
            'R': cam.R.tolist(),
            't': cam.t.tolist(),
            'name': cam.name
        }
        for cam in data['cameras']
    ]

    # Convert numpy arrays to lists for JSON compatibility if needed
    for key in ['Rs_gt', 'ts_gt', 'stick_3d_gt', 'pose_3d_gt', 'all_observed_points_2d',
                'all_pose_projections_2d', 'all_projection_masks', 'K']:
        if key in data_to_save and isinstance(data_to_save[key], np.ndarray):
            data_to_save[key] = data_to_save[key].tolist()
        elif key in data_to_save and isinstance(data_to_save[key], list):
            data_to_save[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr
                               for arr in data_to_save[key]]

    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved synthetic data to {filepath}")
    return str(filepath)

def load_synthetic_data(filepath):
    """
    Load synthetic data from disk

    Args:
        filepath: Path to saved data file

    Returns:
        Dictionary containing synthetic data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Convert back to numpy arrays
    for key in ['Rs_gt', 'ts_gt', 'stick_3d_gt', 'pose_3d_gt', 'all_observed_points_2d',
                'all_pose_projections_2d', 'all_projection_masks', 'K']:
        if key in data:
            data[key] = np.array(data[key])

    # Reconstruct camera objects
    if 'cameras' in data:
        cameras = []
        for cam_data in data['cameras']:
            cam = Camera(
                K=np.array(cam_data['K']),
                R=np.array(cam_data['R']),
                t=np.array(cam_data['t']),
                name=cam_data['name']
            )
            cameras.append(cam)
        data['cameras'] = cameras

    print(f"Loaded synthetic data from {filepath}")
    return data

def plot_2d_trajectories_individual(data, dataset_name="dataset", save_individual=True):
    """Plot 2D trajectories for each camera view and optionally save individual camera plots"""
    if data['num_cameras'] == 0:
        print("No cameras available for plotting")
        return

    all_plots_saved = []

    for cam_idx in range(data['num_cameras']):
        # Get camera info
        camera = data['cameras'][cam_idx]

        # Get 2D projections for this camera
        if cam_idx < len(data['all_observed_points_2d']):
            stick_2d = np.array(data['all_observed_points_2d'][cam_idx])  # [frames, 2_endpoints, xy]
            pose_2d = np.array(data['all_pose_projections_2d'][cam_idx])   # [frames, joints, xy]
            visibility_mask = np.array(data['all_projection_masks'][cam_idx])  # [frames, joints]
        else:
            continue

        # Create individual plot for this camera
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot stick trajectory (both endpoints)
        if stick_2d.shape[0] > 0:
            # Stick start point (red)
            stick_start_x = stick_2d[:, 0, 0]
            stick_start_y = stick_2d[:, 0, 1]
            valid_start = (stick_start_x != 0) | (stick_start_y != 0)

            if np.any(valid_start):
                ax.plot(stick_start_x[valid_start], stick_start_y[valid_start],
                       'r-', linewidth=3, alpha=0.9, label='Stick Start')
                ax.scatter(stick_start_x[valid_start], stick_start_y[valid_start],
                          c='red', s=30, alpha=0.7, zorder=5, edgecolors='darkred')

            # Stick end point (green)
            stick_end_x = stick_2d[:, 1, 0]
            stick_end_y = stick_2d[:, 1, 1]
            valid_end = (stick_end_x != 0) | (stick_end_y != 0)

            if np.any(valid_end):
                ax.plot(stick_end_x[valid_end], stick_end_y[valid_end],
                       'g-', linewidth=3, alpha=0.9, label='Stick End')
                ax.scatter(stick_end_x[valid_end], stick_end_y[valid_end],
                          c='green', s=30, alpha=0.7, zorder=5, edgecolors='darkgreen')

        # Plot human pose trajectories for key joints
        key_joints = {
            'Head': 5,
            'Right Hand': 9,
            'Left Hand': 12,
            'Pelvis': 0,
            'Right Shoulder': 2,
            'Left Shoulder': 3
        }

        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for i, (joint_name, joint_idx) in enumerate(key_joints.items()):
            if joint_idx < pose_2d.shape[1]:
                joint_x = pose_2d[:, joint_idx, 0]
                joint_y = pose_2d[:, joint_idx, 1]

                # Use visibility mask to filter valid points
                if joint_idx < visibility_mask.shape[1]:
                    valid_mask = visibility_mask[:, joint_idx]
                else:
                    valid_mask = (joint_x != 0) | (joint_y != 0)

                if np.any(valid_mask):
                    color = colors[i % len(colors)]
                    ax.plot(joint_x[valid_mask], joint_y[valid_mask],
                           '--', color=color, linewidth=2, alpha=0.8,
                           label=f'{joint_name}')
                    ax.scatter(joint_x[valid_mask], joint_y[valid_mask],
                              c=color, s=20, alpha=0.6, zorder=4)

        # Set axis properties
        ax.set_xlim(0, camera.image_width)
        ax.set_ylim(camera.image_height, 0)  # Invert Y-axis for image coordinates
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(f'{camera.name} - 2D Trajectories\n{dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        # Save individual camera plot
        if save_individual:
            output_filename = f"data/{dataset_name}_{camera.name}_2d_trajectory.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            all_plots_saved.append(output_filename)
            print(f"Saved individual 2D trajectory plot: {output_filename}")

        plt.close()

    return all_plots_saved

def replay_synthetic_data(data, output_path=None, fps=10):
    """
    Replay/visualize saved synthetic data

    Args:
        data: Dictionary containing synthetic data
        output_path: Optional path to save animation
        fps: Frames per second for animation
    """
    print("Replaying synthetic data...")

    # Reconstruct pose frames from 3D ground truth
    pose_frames = []
    for frame_idx in range(data['n_frames']):
        joints_3d = data['pose_3d_gt'][frame_idx]
        pose = HumanPose3D(joints_3d)
        pose_frames.append(pose)

    # Visualize
    visualize_animation(
        cameras=data['cameras'],
        pose_frames=pose_frames,
        output_path=output_path,
        stick_hand_indices=data['hand_indices'],
        stick_length=data['stick_length'],
        two_handed=data['two_handed'],
        fps=fps
    )

def generate_and_plot_synthetic_data(num_cameras=4, n_frames=30, stick_length=1.2,
                                   motion_type='circle', path_size=2.0, noise_std=0.5,
                                   hand_indices=[9], two_handed=False, camera_radius=4.5,
                                   dataset_name="synthetic_dataset", plot_trajectories=True):
    """
    Generate synthetic data and optionally plot 2D trajectories

    This is a convenience function that combines generation and plotting
    """
    # Generate the synthetic data
    data = generate_synthetic_data(
        num_cameras=num_cameras,
        n_frames=n_frames,
        stick_length=stick_length,
        motion_type=motion_type,
        path_size=path_size,
        noise_std=noise_std,
        hand_indices=hand_indices,
        two_handed=two_handed,
        camera_radius=camera_radius
    )

    if plot_trajectories and data['num_cameras'] > 0:
        try:
            # Plot individual camera trajectories
            individual_plots = plot_2d_trajectories_individual(data, dataset_name, save_individual=True)
            print(f"Generated {len(individual_plots)} individual camera trajectory plots")
        except Exception as e:
            print(f"Warning: Failed to plot 2D trajectories: {e}")

    return data

if __name__ == "__main__":
    main()