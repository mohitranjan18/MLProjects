from flask import Flask, render_template, request,send_file
import feature
import pandas as pd
import feature
import numpy as np
from io import BytesIO
import joblib

# create a Flask app variable
app = Flask(__name__,template_folder='templates')

#define a route
@app.route('/')
def index():
	return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
	import feature
	
	if request.method == 'POST':
		file=request.files['file']
		pred_df=pd.read_csv(file)
		main_df=pred_df
		pred_df=feature.data_selection(pred_df,['id_task_session','Remarks on Score','made_submission','js_comments'])
		pred_df=feature.data_validation(pred_df,['tx_difficulty','total_compute','tx_file_type'])
		print('---------------------------')
		# print(pred_df.columns)
		print('---------------------------')
		if 'compile_success' in pred_df.columns:
			pred=feature.data_preprocessing(pred_df,{'compile_success':'int64'})
		pred_df=feature.feature_extractor(pred_df)
		print('file upload successfully')
		data = np.array(pred_df)
		model = joblib.load("../model.pkl")
		pred=model.predict(pred_df)
		result=pd.DataFrame(np.array(pred), columns =['Score'])
		# print(score)
		main_df['score']=result['Score']
		output_file=main_df.to_csv("../result.csv", index=False)
		response_stream = BytesIO(main_df.to_csv().encode())
		return send_file(
				response_stream,
				mimetype="text/csv",
				attachment_filename="../result.csv",
				)
		# return "file uploaded"
	else:
		return render_template('index.html')

if __name__ == '__main__':
	# running app in debug mode
	app.run(debug=True)

