#include "RobustMatcher.h"
using namespace cv;

cv::Mat RobustMatcher::match(cv::Mat& image1, 
                cv::Mat& image2, // input images 
     // output matches and keypoints 
     std::vector<cv::DMatch>& matches, 
     std::vector<cv::KeyPoint>& keypoints1, 
     std::vector<cv::KeyPoint>& keypoints2) { 

   // 1a. Detection of the SURF features 
   detector->detect(image1,keypoints1); 
   detector->detect(image2,keypoints2); 
   // 1b. Extraction of the SURF descriptors 
   cv::Mat descriptors1, descriptors2; 
   extractor->compute(image1,keypoints1,descriptors1); 
   extractor->compute(image2,keypoints2,descriptors2); 
   cv::Mat fundemental;
   
   // 2. Match the two image descriptors 
   // Construction of the matcher 
   //cv::BruteForceMatcher<cv::L2<float>> matcher; 
   // from image 1 to image 2 
   // based on k nearest neighbours (with k=2) 
   std::vector<std::vector<cv::DMatch> > matches1; 
   matcher->knnMatch(descriptors1,descriptors2, 
       matches1, // vector of matches (up to 2 per entry) 
       2);        // return 2 nearest neighbours 
   
    // from image 2 to image 1 
    // based on k nearest neighbours (with k=2) 
    std::vector<std::vector<cv::DMatch> > matches2; 
    matcher->knnMatch(descriptors2,descriptors1, 
       matches2, // vector of matches (up to 2 per entry) 
       2);        // return 2 nearest neighbours 

	
    // 3. Remove matches for which NN ratio is 
    // > than threshold 
    // clean image 1 -> image 2 matches 

    int removed= ratioTest(matches1); 
    // clean image 2 -> image 1 matches 
    removed= ratioTest(matches2); 
    // 4. Remove non-symmetrical matches 
    std::vector<cv::DMatch> symMatches; 
    symmetryTest(matches1,matches2,symMatches); 
    // 5. Validate matches using RANSAC 
    fundemental= ransacTest(symMatches, 
                keypoints1, keypoints2, matches); 

	
    // return the found fundemental matrix 
    return fundemental; 
  } 


cv::Mat RobustMatcher::match(//KeyPoints and descriptors are already computed
     std::vector<cv::DMatch>& matches, 
     std::vector<cv::KeyPoint>& keypoints1, 
     std::vector<cv::KeyPoint>& keypoints2,
	 cv::Mat& descriptors1,
	 cv::Mat& descriptors2
	 )
{ 

   // 2. Match the two image descriptors 
   // Construction of the matcher 
   // from image 1 to image 2 
   // based on k nearest neighbours (with k=2) 
   std::vector<std::vector<cv::DMatch> > matches1; 
   matcher->knnMatch(descriptors1,descriptors2, 
       matches1, // vector of matches (up to 2 per entry) 
       2);        // return 2 nearest neighbours 
    // from image 2 to image 1 
    // based on k nearest neighbours (with k=2) 
    std::vector<std::vector<cv::DMatch> > matches2; 
    matcher->knnMatch(descriptors2,descriptors1, 
       matches2, // vector of matches (up to 2 per entry) 
       2);        // return 2 nearest neighbours 
    // 3. Remove matches for which NN ratio is 
    // > than threshold 
    // clean image 1 -> image 2 matches 
    int removed= ratioTest(matches1); 
    // clean image 2 -> image 1 matches 
    removed= ratioTest(matches2); 
    // 4. Remove non-symmetrical matches 
    std::vector<cv::DMatch> symMatches; 
    symmetryTest(matches1,matches2,symMatches); 
    // 5. Validate matches using RANSAC 
    cv::Mat fundemental= ransacTest(symMatches, 
                keypoints1, keypoints2, matches); 
    // return the found fundemental matrix 
    return fundemental; 
  } 