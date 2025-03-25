import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QFileDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class HedgehogVisualizer3D(QMainWindow):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.init_ui()
        self.load_data()
        self.setup_controls()
        self.update_plot()

    def init_ui(self):
        self.setWindowTitle("Velocity Hedgehog 3D Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout(main_widget)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 85)

        self.control_panel = QWidget()
        control_layout = QHBoxLayout(self.control_panel)

        # z controller
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_label = QLabel("Z: 0.00m")
        control_layout.addWidget(QLabel("Height (Z):"))
        control_layout.addWidget(self.z_slider)
        control_layout.addWidget(self.z_label)

        # distance controller
        self.dis_slider = QSlider(Qt.Horizontal)
        self.dis_label = QLabel("Distance: 0.00m")
        control_layout.addWidget(QLabel("Distance:"))
        control_layout.addWidget(self.dis_slider)
        control_layout.addWidget(self.dis_label)

        self.layout.addWidget(self.control_panel, 15)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        export_action = file_menu.addAction('Export Plot')
        export_action.triggered.connect(self.export_plot)

    def load_data(self):

        self.Z = np.load(f"{self.data_path}/robot_zs.npy")
        self.Dis = np.load(f"{self.data_path}/robot_diss.npy")
        self.Phi = np.load(f"{self.data_path}/robot_phis.npy")
        self.Gamma = np.load(f"{self.data_path}/robot_gammas.npy")
        self.vel_max = np.load(f"{self.data_path}/z_dis_phi_gamma_vel_max.npy")

        self.vmax = np.nanmax(self.vel_max)

    def setup_controls(self):
        self.z_slider.setRange(0, len(self.Z) - 1)
        self.dis_slider.setRange(0, len(self.Dis) - 1)
        self.z_slider.valueChanged.connect(self.update_plot)
        self.dis_slider.valueChanged.connect(self.update_plot)

    def update_plot(self):
        z_idx = self.z_slider.value()
        dis_idx = self.dis_slider.value()

        self.z_label.setText(f"Z: {self.Z[z_idx]:.2f}m")
        self.dis_label.setText(f"Distance: {self.Dis[dis_idx]:.2f}m")

        # velocity when fixing z and distance
        speed_slice = self.vel_max[z_idx, dis_idx, :, :]

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        Gamma_grid, Phi_grid = np.meshgrid(self.Gamma, self.Phi)

        surf = ax.plot_surface(
            Gamma_grid, Phi_grid, speed_slice,
            cmap='viridis',
            edgecolor='none',
            vmin=0,
            vmax=self.vmax
        )

        self.figure.colorbar(surf, ax=ax, label='Velocity (m/s)', shrink=0.5, aspect=10)

        ax.set_title(f"Max Velocity at Z={self.Z[z_idx]:.2f}m, Distance={self.Dis[dis_idx]:.2f}m")
        ax.set_xlabel('Gamma (rad)')
        ax.set_ylabel('Phi (rad)')
        ax.set_zlabel('Velocity (m/s)')

        ax.view_init(elev=30, azim=45)

        self.canvas.draw()

    def export_plot(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "", "PNG Files (*.png)", options=options)

        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
            self.figure.savefig(filename, dpi=300)


class LandingVisualizer3D(QMainWindow):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.init_ui()
        self.load_data()
        self.setup_controls()
        self.update_plot()

    def init_ui(self):
        self.setWindowTitle("Projectile Velocity and Range Visualizer")
        self.setGeometry(100, 100, 1280, 900)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout(main_widget)

        # Create 3D canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 85)

        # Control panel
        self.control_panel = QWidget()
        control_layout = QHBoxLayout(self.control_panel)

        # Height control components
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_label = QLabel("Initial Height: 0.00m")
        control_layout.addWidget(QLabel("Launch Height:"))
        control_layout.addWidget(self.z_slider)
        control_layout.addWidget(self.z_label)

        # Distance control components
        self.dis_slider = QSlider(Qt.Horizontal)
        self.dis_label = QLabel("Target Distance: 0.00m")
        control_layout.addWidget(QLabel("Target Distance:"))
        control_layout.addWidget(self.dis_slider)
        control_layout.addWidget(self.dis_label)

        self.layout.addWidget(self.control_panel, 15)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        export_action = file_menu.addAction('Export Plot')
        export_action.triggered.connect(self.export_plot)

    def load_data(self):
        """Load precomputed parameter grids and velocity data"""
        self.Z = np.load(f"{self.data_path}/robot_zs.npy")  # Height grid
        self.Dis = np.load(f"{self.data_path}/robot_diss.npy")  # Distance grid
        self.Phi = np.load(f"{self.data_path}/robot_phis.npy")  # Pitch angle
        self.Gamma = np.load(f"{self.data_path}/robot_gammas.npy")  # Yaw angle
        self.vel_max = np.load(f"{self.data_path}/z_dis_phi_gamma_vel_max.npy")  # Max velocity data

        # Initialize data container for flight distance
        self.flight_distance = np.zeros_like(self.vel_max)

    def setup_controls(self):
        """Initialize control ranges"""
        self.z_slider.setRange(0, len(self.Z) - 1)
        self.dis_slider.setRange(0, len(self.Dis) - 1)
        self.z_slider.valueChanged.connect(self.update_plot)
        self.dis_slider.valueChanged.connect(self.update_plot)

    def calculate_flight_distance(self, z0, vel, gamma):
        """Calculate flight distance based on initial height, velocity, and angle"""
        vz0 = vel * np.sin(gamma)  # Vertical velocity component
        vx0 = vel * np.cos(gamma)  # Horizontal velocity component

        # Calculate flight time
        discriminant = vz0 ** 2 + 2 * 9.81 * (z0-0)
        if discriminant < 0:
            return np.nan  # No solution if discriminant is negative

        t_flight = (vz0 + np.sqrt(discriminant)) / 9.81
        return vx0 * t_flight  # Horizontal distance

    def update_plot(self):
        """Dynamically update the 3D plot"""
        z_idx = self.z_slider.value()
        dis_idx = self.dis_slider.value()
        current_z = self.Z[z_idx]
        current_dis = self.Dis[dis_idx]

        # Update labels
        self.z_label.setText(f"Launch Height: {current_z:.2f}m")
        self.dis_label.setText(f"Target Distance: {current_dis:.2f}m")

        # Extract velocity slice for current z and dis
        velocity_slice = self.vel_max[z_idx, dis_idx]

        # Calculate flight distance for each gamma and phi
        for i, phi in enumerate(self.Phi):
            for j, gamma in enumerate(self.Gamma):
                self.flight_distance[z_idx, dis_idx, i, j] = self.calculate_flight_distance(
                    current_z, velocity_slice[i, j], gamma
                )

        # Handle all-NaN case
        self.figure.clear()
        if np.all(np.isnan(self.flight_distance[z_idx, dis_idx])):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No solution for current parameters\nAdjust height or distance',
                    ha='center', va='center', fontsize=16)
            self.canvas.draw()
            return

        # Create 3D surface
        ax = self.figure.add_subplot(111, projection='3d')
        Gamma_grid, Phi_grid = np.meshgrid(self.Gamma, self.Phi)

        # Filter invalid values
        valid_mask = ~np.isnan(self.flight_distance[z_idx, dis_idx])
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            self.canvas.draw()
            return

        # Plot surface
        surf = ax.plot_surface(
            Gamma_grid, Phi_grid, self.flight_distance[z_idx, dis_idx],
            cmap='viridis',
            edgecolor='none',
            vmin=np.nanmin(self.flight_distance[z_idx, dis_idx]),
            vmax=np.nanmax(self.flight_distance[z_idx, dis_idx])
        )

        # Add color bar
        self.figure.colorbar(surf, ax=ax, label='Flight Distance (m)', shrink=0.6)

        # Set labels and title
        ax.set_title(
            f"Flight Distance (Height={current_z:.1f}m, Target Distance={current_dis:.1f}m)\n"
            f"Velocity: {np.nanmax(velocity_slice):.2f}m/s"
        )
        ax.set_xlabel('Gamma (rad)')
        ax.set_ylabel('Phi (rad)')
        ax.set_zlabel('Flight Distance (m)')

        # Adjust view angle
        ax.view_init(elev=30, azim=-45)
        self.canvas.draw()

    def export_plot(self):
        """Export the current plot to an image file"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png)", options=options)
        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
            self.figure.savefig(filename, dpi=300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    PATH = '../hedgehog_data'
    # PATH = '../../../mobile-throwing/robot_data/panda_5_joint_fix_0.3'

    Hedgehog_viewer = HedgehogVisualizer3D(PATH)
    Hedgehog_viewer.show()

    # Landing_viewer = LandingVisualizer3D(PATH)
    # Landing_viewer.show()

    sys.exit(app.exec_())