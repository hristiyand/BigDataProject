import numpy as np
from PIL import Image
import numpy
import open3d as o3d
import laspy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def laspy_writer_rgb_depth(coordinates, rgb, intensity, path: str) -> None:
    # writes a .las format point cloud given point coordinates, colour values and intensities

    # define empty las
    hdr = laspy.header.Header(file_version=1.4, point_format=7)
    mins = np.floor(np.min(xyz, axis=1))
    hdr.offset = mins
    outfile = laspy.file.File(path, mode="w", header=hdr)
    outfile.header.scale = [0.01, 0.01, 0.01]

    # fill in with data
    outfile.X = coordinates[:,0]
    outfile.Y = coordinates[:,1]
    outfile.Z = coordinates[:,2]
    outfile.Red = rgb[:,0]
    outfile.Green = rgb[:,1]
    outfile.Blue = rgb[:,2]
    outfile.Intensity = intensity
    outfile.close()


def laspy_reader_rgb_depth(path: str):
    # reads a .las point cloud

    # reading las file and copy points
    input_las = laspy.file.File(path, mode="r")
    point_records = input_las.points.copy()

    # getting scaling and offset parameters
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)

    # plotting points
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(p_X, p_Y, p_Z, c='r', marker='o')
    #plt.show()


def ply_writer(points: np.ndarray, path: str) -> None:
    # Writes a .ply format point cloud

    pcd = o3d.geometry.PointCloud()
    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(points)
    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)


# extracts data from .tif imagery (currently dsm
im_depth = Image.open("D:\\Desktop\\UNI\\Master\\Masterarbeit\\Python\\Images\\UH_NAD83_271460_3289689_dsm.tif")
im_rgb = Image.open("D:\\Desktop\\UNI\\Master\\Masterarbeit\\Python\\Images\\UH_NAD83_271460_3289689_input.tif")
depth_array = np.array(im_depth, dtype=float)
rgb_array = np.array(im_rgb, dtype='uint8')

# define point cloud as array
xyz = np.ndarray(shape=(1202,1192,3), dtype=float)
i = depth_array.flatten()
# fill in with data
for x in range(0,1202):
    for y in range(0,1192):
              xyz[x,y] = [x,y,depth_array[x,y]]
# reshape to write point cloud files
points = xyz.reshape((xyz.shape[0]*xyz.shape[1]),xyz.shape[2])
colour = rgb_array.reshape((rgb_array.shape[0]*rgb_array.shape[1]),rgb_array.shape[2])

# write las
dir = "D:\\Desktop\\UNI\\Master\\Masterarbeit\\BigData\\Test_PC.las"
laspy_writer_rgb_depth(points, colour, i, dir)
# write ply
dir2 = "D:/Desktop/UNI/Master/Masterarbeit/BigData/Pythonmy_pts2.ply"

ply_writer(points, dir2)
# read ply file
pcd = o3d.io.read_point_cloud("D:/Desktop/UNI/Master/Masterarbeit/BigData/Pythonmy_pts.ply")
# visualize
o3d.visualization.draw_geometries([pcd])

