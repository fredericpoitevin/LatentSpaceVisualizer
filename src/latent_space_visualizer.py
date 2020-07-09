import numpy as np

from bokeh import events
from bokeh.io import push_notebook, output_notebook, show
from bokeh.layouts import row
from bokeh.models import CustomJS, Div
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

def visualize(dataset_file, image_type, latent_method, latent_idx_1, latent_idx_2, x_axis_label_text_font_size='20pt', y_axis_label_text_font_size='20pt', index_label_text_font_size='20px', image_brightness=1.0, figure_width = 450,
    figure_height = 450, image_size_scale_factor = 0.9):
    with h5.File(dataset_file, "r") as dataset_file_handle:
        images = dataset_file_handle[image_type][:]
        latent = dataset_file_handle[latent_method][:]
        labels = np.zeros(len(images)) # unclear on how to plot targets

    n_labels = len(np.unique(labels))
    
    x = latent[:, latent_idx_1]
    y = latent[:, latent_idx_2]
    
    bit_depth_scale_factor = 255
    thumbnails = get_thumbnails(images, bit_depth_scale_factor)
    
    def get_colors(palette):
        def map_label_to_color(label):
            return viridis_palette[label]
        return list(map(map_label_to_color, labels.astype(np.int)))

    viridis_palette = bokeh.palettes.viridis(n_labels)
    colors = get_colors(viridis_palette)
    
    if latent_method == "principal_component_analysis":
        x_axis_label = "PC {}".format(latent_idx_1 + 1)
        y_axis_label = "PC {}".format(latent_idx_2 + 1)
    elif latent_method == "diffusion_map":
        x_axis_label = "DC {}".format(latent_idx_1 + 1)
        y_axis_label = "DC {}".format(latent_idx_2 + 1)
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")

    p = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    p.scatter(x, y, fill_color=colors, fill_alpha=0.6, line_color=None)
    p.xaxis.axis_label = x_axis_label
    p.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
    p.yaxis.axis_label = y_axis_label
    p.yaxis.axis_label_text_font_size = y_axis_label_text_font_size

    div = Div(width=int(figure_width*image_size_scale_factor), height=int(figure_height*image_size_scale_factor))

    layout = row(p, div)

    point_attributes = ['x', 'y']
    p.js_on_event(events.MouseMove, display_event(div, x, y, thumbnails, image_brightness, attributes=point_attributes, style='font-size:{};text-align:center'.format(index_label_text_font_size)))
    #p.js_on_event(events.Tap, display_event(div, x, y, thumbnails, attributes=point_attributes))

    show(layout)