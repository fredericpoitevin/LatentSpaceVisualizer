import numpy as np

from bokeh import events
import bokeh.io
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.layouts import row
from bokeh.models import CustomJS, Div, LinearColorMapper, ColorBar
from bokeh.plotting import ColumnDataSource
import bokeh.palettes

from kora.bokeh import figure

from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA

from PIL import Image
import base64
from io import BytesIO

import h5py as h5


def get_color(x, color_bar_palette, vmin, vmax):
    n = len(color_bar_palette)
    return color_bar_palette[int((x - vmin) / (vmax - vmin) * n)]

def angle_axis_representation(quaternion):
    q_r = quaternion[0]
    q_ijk = quaternion[1:]
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    q_norm = np.linalg.norm(q_ijk)
    axis = q_ijk / q_norm
    theta = 2 * np.arctan2(q_norm, q_r)
    return axis, theta

def azimuth_elevation_representation(unit_vector):
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return azimuth, elevation

def get_elevation_azimuth_rotation_angles_from_orientations(orientations):
    x = np.zeros((orientations.shape[0],))
    y = np.zeros((orientations.shape[0],))
    z = np.zeros((orientations.shape[0],))
    
    for orientation_idx, orientation in enumerate(orientations):
        axis, theta = angle_axis_representation(orientation)
        azimuth, elevation = azimuth_elevation_representation(axis)

        x[orientation_idx] = azimuth
        y[orientation_idx] = elevation
        z[orientation_idx] = theta
    
    return x, y, z

def get_colors_from_rotation_angles(rotation_angles, color_bar_palette=bokeh.palettes.plasma(256)):
    color_bar_vmin = 0.0
    color_bar_vmax = 2*np.pi
        
    colors = []
    for rotation_angle in rotation_angles:
        color = get_color(rotation_angle, color_bar_palette, color_bar_vmin, color_bar_vmax)
        colors.append(color)
    
    color_mapper = LinearColorMapper(palette=color_bar_palette, low=color_bar_vmin, high=color_bar_vmax)
    return colors, color_mapper

def gnp2im(image_np, bit_depth_scale_factor):
    """
    Converts an image stored as a 2-D grayscale Numpy array into a PIL image.
    
    Assumes values in image_np are between [0, 1].
    """
    return Image.fromarray((image_np * bit_depth_scale_factor).astype(np.uint8), mode='L')

def to_base64(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

def get_thumbnails(data, bit_depth_scale_factor):
    thumbnails = []
    for gnp in data:
        im = gnp2im(gnp, bit_depth_scale_factor)
        memout = BytesIO()
        im.save(memout, format='png')
        thumbnails.append(to_base64(memout.getvalue()))
    return thumbnails

def display_event(div, x, y, thumbnails, image_brightness, attributes=[], style = 'font-size:20px;text-align:center'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div, x=x, y=y, thumbnails=thumbnails, image_brightness=image_brightness), code="""
        var attrs = %s; var args = []; var n = x.length;
        
        var test_x;
        var test_y;
        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] == 'x') {
                test_x = Number(cb_obj[attrs[i]]);
            }
            
            if (attrs[i] == 'y') {
                test_y = Number(cb_obj[attrs[i]]);
            }
        }
    
        var minDiffIndex = -1;
        var minDiff = 99999;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (test_x - x[i]) ** 2 + (test_y - y[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
        
        var img_tag_attrs = "style='filter: brightness(" + image_brightness + ");'";
        var img_tag = "<div><img src='" + thumbnails[minDiffIndex] + "' " + img_tag_attrs + "></img></div>";
        //var line = img_tag + "\\n";
        var line = img_tag + "<p style=%r>" + (minDiffIndex+1) + "</p>" + "\\n";
        div.text = "";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))

def visualize(dataset_file, image_type, latent_method, 
              latent_idx_1=None, latent_idx_2=None, 
              x_axis_label_text_font_size='20pt', y_axis_label_text_font_size='20pt', index_label_text_font_size='20px',
              image_brightness=1.0, 
              figure_width = 450, figure_height = 450, 
              image_size_scale_factor = 0.9, 
              color_bar_height = 400, color_bar_width = 120):
    with h5.File(dataset_file, "r") as dataset_file_handle:
        images = dataset_file_handle[image_type][:]
        latent = dataset_file_handle[latent_method][:]
        labels = np.zeros(len(images)) # unclear on how to plot targets

    n_labels = len(np.unique(labels))
    
    bit_depth_scale_factor = 255
    thumbnails = get_thumbnails(images, bit_depth_scale_factor)
    
    p = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    p.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
    p.yaxis.axis_label_text_font_size = y_axis_label_text_font_size

    div = Div(width=int(figure_width*image_size_scale_factor), height=int(figure_height*image_size_scale_factor))

    if latent_method == "principal_component_analysis":
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        p.scatter(x, y, fill_alpha=0.6)
        p.xaxis.axis_label = "PC {}".format(latent_idx_1 + 1)
        p.yaxis.axis_label = "PC {}".format(latent_idx_2 + 1)

        layout = row(p, div)
    elif latent_method == "diffusion_map":  
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        p.scatter(x, y, fill_alpha=0.6)
        p.xaxis.axis_label = "DC {}".format(latent_idx_1 + 1)
        p.yaxis.axis_label = "DC {}".format(latent_idx_2 + 1)

        layout = row(p, div)
    elif latent_method == "orientations":    
        x, y, rotation_angles = get_elevation_azimuth_rotation_angles_from_orientations(latent)
        
        colors, color_mapper = get_colors_from_rotation_angles(rotation_angles)
                
        p.scatter(x, y, fill_alpha=0.6, fill_color=colors, line_color=None)
        p.xaxis.axis_label = "Azimuth"
        p.yaxis.axis_label = "Elevation"
        
        color_bar_plot = figure(title="Rotation", title_location="right", 
                                height=color_bar_height, width=color_bar_width, 
                                min_border=0, 
                                outline_line_color=None,
                                toolbar_location=None)
        
        color_bar_plot.title.align = "center"
        color_bar_plot.title.text_font_size = "12pt"
        color_bar_plot.scatter([], []) # removes Bokeh warning 1000 (MISSING_RENDERERS)
        
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0,0))
        color_bar_plot.add_layout(color_bar, "right")
        
        layout = row(p, color_bar_plot, div)
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")

    point_attributes = ['x', 'y']
    p.js_on_event(events.MouseMove, display_event(div, x, y, thumbnails, image_brightness, attributes=point_attributes, style='font-size:{};text-align:center'.format(index_label_text_font_size)))
    #p.js_on_event(events.Tap, display_event(div, x, y, thumbnails, attributes=point_attributes))

    show(layout)

def output_notebook():
	bokeh.io.output_notebook()
