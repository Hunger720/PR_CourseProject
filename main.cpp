#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void main(){
	CvBoostParams params;
	CvBoost model;

	int train_pos_number = 672+1, train_neg_number = 4843+1;
	int test_pos_number = 1126+1, test_neg_number = 1527+1, test_total_number = test_pos_number + test_neg_number;
	int feature_dimensions = 2330;
	float count = 0;
	string train_path = "D:\\用户目录\\我的文档\\SCUT\\First Year\\Second Semester\\Courses\\Pattern Recognition\\PR_CourseProject_data\\train\\";
	string test_path = "D:\\用户目录\\我的文档\\SCUT\\First Year\\Second Semester\\Courses\\Pattern Recognition\\PR_CourseProject_data\\test\\";
	string number;
	stringstream file_number;
	ifstream data_file;

	Mat trainData(train_pos_number+train_neg_number,feature_dimensions,CV_32FC1);
	Mat responses(train_pos_number+train_neg_number,1,CV_32FC1);
	Mat testData(test_pos_number+test_neg_number,feature_dimensions,CV_32FC1);
	Mat resp(test_pos_number+test_neg_number,1,CV_32FC1);
	Mat results(test_pos_number+test_neg_number,1,CV_32FC1);


	//1.Read data files.
	//Training dataset.
	for(int i = 0;i<train_neg_number;i++){
		file_number << i;
		data_file.open(train_path+"neg\\"+file_number.str()+".txt");
		cout<<"Loading train data file neg"<<file_number.str()<<endl;
		file_number.str("");
		for(int j = 0;j<feature_dimensions;j++){
			data_file >> number;
			trainData.at<float>(i,j) = atof(number.c_str());
		}
		responses.at<float>(i,0) = -1;
		data_file.close();
	}
	for(int i = 0;i<train_pos_number;i++){
		file_number << i;
		data_file.open(train_path+"pos\\"+file_number.str()+".txt");
		cout<<"Loading train data file pos"<<file_number.str()<<endl;
		file_number.str("");
		for(int j = 0;j<feature_dimensions;j++){
			data_file >> number;
			trainData.at<float>(i+train_neg_number,j) = atof(number.c_str());
		}
		responses.at<float>(i+train_neg_number,0) = 1;
		data_file.close();
	}
	//Testing dataset.
	for(int i = 0;i<test_neg_number;i++){
		file_number << i;
		data_file.open(test_path+"neg\\"+file_number.str()+".txt");
		cout<<"Loading test data file neg"<<file_number.str()<<endl;
		file_number.str("");
		for(int j = 0;j<feature_dimensions;j++){
			data_file >> number;
			testData.at<float>(i,j) = atof(number.c_str());
		}
		resp.at<float>(i,0) = -1;
		data_file.close();
	}
	for(int i = 0;i<test_pos_number;i++){
		file_number << i;
		data_file.open(test_path+"pos\\"+file_number.str()+".txt");
		cout<<"Loading test data file pos"<<file_number.str()<<endl;
		file_number.str("");
		for(int j = 0;j<feature_dimensions;j++){
			data_file >> number;
			testData.at<float>(i+test_neg_number,j) = atof(number.c_str());
		}
		resp.at<float>(i+test_neg_number,0) = 1;
		data_file.close();
	}


	//2.Train and predict.
	cout<<"Training model ...";
	model.train(trainData, CV_ROW_SAMPLE, responses);
	cout<<"done!"<<endl<<"Testing ...";
	for(int i = 0;i<test_pos_number+test_neg_number;i++){
		results.at<float>(i,0) = model.predict(testData.row(i));
		if(results.at<float>(i,0) == resp.at<float>(i,0)) count++;
	}
	cout<<"done!"<<endl;

	//3.Evaluating the results.
	cout<<"Precesion: "<<count/test_total_number<<endl;
}