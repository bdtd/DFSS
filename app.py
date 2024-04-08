from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

#dic = {0 : 'cat', 1 : 'dog', 2 :"horse", 3:'human'}

#def contrastive_loss(y_true, y_pred):
#    """Calculates the contrastive loss.
#
#    Arguments:
#        y_true: List of labels, each label is of type float32.
#        y_pred: List of predictions of same length as of y_true,
#                each label is of type float32.
#
#    Returns:
#        A tensor containing contrastive loss as floating point value.
#    """
#
#    square_pred = tf.math.square(y_pred)
#    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#    loss = tf.math.reduce_mean(
#        (1 - y_true) * square_pred + (y_true) * margin_square
#    )
#    return loss

#model = load_model('siamese_mobilenet_crop2.hdf5', 
#                    custom_objects={'contrastive_loss': contrastive_loss})

#model._make_predict_function()
interpreter = tflite.Interpreter('quan_siamese_mobilenet_crop_7e.tflite')



def predict_label(img_path):
	i = Image.open(img_path)
	target_size = (224, 224)
	i = i.resize(target_size)
	i = np.asarray(i).astype('float32')
	i = i.reshape(1, 224,224,3)

#	j = Image.open("/media/junaid/DATA/Gurjot/cedar_crop/processed_forg/1/forgeries_1_1.png")
#	target_size = (224, 224)
#	j = j.resize(target_size)
#	j = np.asarray(j).astype('float32')
#	j = j.reshape(1, 224,224,3)
#	j = image.load_img("/media/junaid/DATA/Gurjot/cedar_crop/processed_forg/1/forgeries_1_1.png", target_size=(224,224))
#	j = image.img_to_array(j)
#	j = j.reshape(1, 224,224,3)
	imgs = np.concatenate((i, i), axis=3)
#	p = model.predict(imgs)
	interpreter.allocate_tensors()
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

    # interpreter.set_tensor(input_index, np.moveaxis(test_images, 3, 1))
	interpreter.set_tensor(input_index, imgs)
    # Run inference.
	interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
	output = interpreter.get_tensor(output_index)
	return np.squeeze(output) * 100


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("index.html", prediction = p, img_path = img_path)



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

