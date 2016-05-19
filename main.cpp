#include <fstream>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void main(){
	CvBoostParams params;
	CvBoost classifier;
	CvBoost classifier1;
	CvBoost classifier2;

	int train_pos_number = 672+1, train_neg_number = 4843+1, train_total_number = train_pos_number + train_neg_number;
	int test_pos_number = 1126+1, test_neg_number = 1527+1, test_total_number = test_pos_number + test_neg_number;
	int feature_dimensions = 2330, weak_count = 120;
	int sub1_number = train_total_number/2, sub2_number = train_total_number - sub1_number;
	int* index = new int[train_total_number];
	float count = 0;
	float* results;
	string train_path = "D:\\用户目录\\我的文档\\SCUT\\First Year\\Second Semester\\Courses\\Pattern Recognition\\PR_CourseProject_data\\train\\";
	string test_path = "D:\\用户目录\\我的文档\\SCUT\\First Year\\Second Semester\\Courses\\Pattern Recognition\\PR_CourseProject_data\\test\\";
	string number;
	stringstream file_number;
	ifstream data_file;

	clock_t start, end;

	Mat trainData(train_pos_number+train_neg_number,feature_dimensions,CV_32FC1);
	Mat responses(train_pos_number+train_neg_number,1,CV_32FC1);
	Mat testData(test_pos_number+test_neg_number,feature_dimensions,CV_32FC1);
	Mat resp(test_pos_number+test_neg_number,1,CV_32FC1);
	Mat sub1_data, sub2_data, sub1_resp, sub2_resp;

	params.weak_count = weak_count;


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
	cout<<endl<<"Training classifier using the whole training dataset ...";
	start = clock();
	classifier.train(trainData, CV_ROW_SAMPLE, responses, Mat(), Mat(), Mat(), Mat(), params);
	end = clock();
	cout<<"done!"<<endl
		<<"Training time: "<<(double)(end - start)/CLOCKS_PER_SEC<<'s'<<endl
		<<"Testing on test dataset ...";
	results = new float[test_total_number];
	start = clock();
	for(int i = 0;i<test_total_number;i++){
		results[i] = classifier.predict(testData.row(i));
		if(results[i] == resp.at<float>(i,0)) count++;
	}
	end = clock();
	cout<<"done!"<<endl
		<<"Test time; "<<(double)(end - start)/CLOCKS_PER_SEC<<'s'<<endl;


	//3.Evaluating the results.
	cout<<"Accuracy: "<<count/test_total_number<<endl;


	//4.Run 2-fold cross-validation.
	//Randomly devide the training dataset into 2 subsets.
	for(int i = 0;i < train_total_number;i++)
		index[i] = i;
	srand(time(NULL));
	int last = train_total_number, r;
	for(int i = 0;i < sub1_number; i++){
		r = rand()%last;
		sub1_data.push_back(trainData.row(index[r]));
		sub1_resp.push_back(responses.row(index[r]));
		index[r] = index[--last];
	}
	for(int i = 0;i < sub2_number; i++){
		r = rand()%last;
		sub2_data.push_back(trainData.row(index[r]));
		sub2_resp.push_back(responses.row(index[r]));
		index[r] = index[--last];
	}
	//Train and predict for the first time.
	cout<<endl<<"Training classifier using subset1 ...";
	start = clock();
	classifier1.train(sub1_data, CV_ROW_SAMPLE, sub1_resp, Mat(), Mat(), Mat(), Mat(), params);
	end = clock();
	cout<<"done!"<<endl
		<<"Training time: "<<(double)(end - start)/CLOCKS_PER_SEC<<'s'<<endl
		<<"Testing on subset2 ...";
	count = 0;
	delete results; results = new float[sub2_number];
	start = clock();
	for(int i = 0;i < sub2_number; i++){
		results[i] = classifier1.predict(sub2_data.row(i));
		if(results[i] == sub2_resp.at<float>(i,0)) count++;
	}
	end = clock();
	cout<<"done!"<<endl
		<<"Testing time: "<<(double)(end-start)/CLOCKS_PER_SEC<<'s'<<endl;
	//Evaluating the results.
	cout<<"Accuracy: "<<count/sub2_number<<endl;

	//Train and predict for the second time.
	cout<<endl<<"Training classifier using subset2 ...";
	start = clock();
	classifier2.train(sub2_data, CV_ROW_SAMPLE, sub2_resp, Mat(), Mat(), Mat(), Mat(), params);
	end = clock();
	cout<<"done!"<<endl
		<<"Testing time: "<<(double)(end - start)/CLOCKS_PER_SEC<<'s'<<endl
		<<"Testing on subset1 ...";
	count = 0;
	delete results; results = new float[sub1_number];
	start = clock();
	for(int i = 0;i < sub1_number; i++){
		results[i] = classifier2.predict(sub1_data.row(i));
		if(results[i] == sub1_resp.at<float>(i,0)) count++;
	}
	end = clock();
	cout<<"done!"<<endl
		<<"Testing time: "<<(double)(end - start)/CLOCKS_PER_SEC<<'s'<<endl;
	//Evaluating the results.
	cout<<"Accuracy: "<<count/sub1_number<<endl;
}