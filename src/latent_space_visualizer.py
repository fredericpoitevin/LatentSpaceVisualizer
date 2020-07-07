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

def display_event(div, x, y, thumbnails, figure_width, figure_height, image_brightness, attributes=[], style = 'float:left;clear:left;font_size=13px'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div, x=x, y=y, thumbnails=thumbnails, figure_width=figure_width, figure_height=figure_height, image_brightness=image_brightness), code="""
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
        
        var img_tag_attrs = "height='" + (figure_height) + "' width='" + (figure_width) + "' style='float: left; margin: 0px 15px 15px 0px; filter: brightness(" + image_brightness + ");' border='2'";
        var img_tag = "<div><img src='" + thumbnails[minDiffIndex] + "' " + img_tag_attrs + "></img></div>";
        var line = "<span style=%r>Index: " + minDiffIndex + "</span>" + img_tag + "\\n";
        div.text = "";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))

def load_noise_dataset(n, h, w):
    images = np.random.rand(n, h, w)
    pca = PCA(n_components=1)
    z = pca.fit_transform(images.reshape((n, h * w)))
    labels = np.zeros(len(images))
    labels[z.flatten() <= 0] = 0
    labels[z.flatten() > 0] = 1
    noise_dataset = {
        'images': images,
        'target': labels
    }
    return noise_dataset

def load_diffraction_dataset(dataset_filepath, n_pca_components=2):
    images = np.load(dataset_filepath)
    n, h, w = images.shape
    pca = PCA(n_components=n_pca_components)
    scaled_image_vectors = preprocessing.minmax_scale(images.reshape((n, h * w)), feature_range=(0, 1))
    scaled_images = scaled_image_vectors.reshape((n, h, w))
    latent = pca.fit_transform(scaled_image_vectors)
    labels = np.zeros(len(images))
    diffraction_dataset = {
      'images': scaled_images,
      'latent': latent,
      'target': labels
    }
    return diffraction_dataset

def visualize(dataset, latent_idx_1, latent_idx_2, image_brightness=1.0):
    images = dataset['images']
    latent = dataset['latent']
    labels = dataset['target']
    
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

    figure_width = 450
    figure_height = 450

    p = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    p.scatter(x, y, fill_color=colors, fill_alpha=0.6, line_color=None)

    div = Div()

    layout = row(p, div)

    point_attributes = ['x', 'y']
    p.js_on_event(events.MouseMove, display_event(div, x, y, thumbnails, figure_width, figure_height, image_brightness, attributes=point_attributes))
    #p.js_on_event(events.Tap, display_event(div, x, y, thumbnails, figure_width, figure_height, attributes=point_attributes))

    show(layout)