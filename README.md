*******************************************************************************************************
  In this version, utilizing the tools in scikit-learn, we attempt to build our experimental platform in a more convinient way. Specifically, the contributions of this new
version are:
1. We could automatically tune the parameters which are pre-designed by our team members.
2. A unified procedure to write the modules in the RecommendationAlg package is clearly defined.
  When implementing the recommendation algorithms, we should first inherit the base class of BaseEstimator, and then implement at least three functions including
  predict, fit and score. The declarations of these functions are shown as below:
  (1)def predict(self,testSamples)
  (2)def fit(self, trainSamples, trainTargets)
  (3)def score(self, testSamples, trueLabels)


*******************************************************************************************************
# Recommender

## 提交流程
查看文件状态

git status

跟踪新文件

git add [file|.]

提交

git commit –m "说明"

上传服务器master分支

git push origin master

## 同步流程
获取更新

git fetch

同步本地和master

git rebase


