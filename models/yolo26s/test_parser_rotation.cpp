#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"
extern "C" bool NvDsInferParseYolo(const std::vector<NvDsInferLayerInfo>&,
 const NvDsInferNetworkInfo&, const NvDsInferParseDetectionParams&,
 std::vector<NvDsInferParseObjectInfo>&);
int main() {
 float data[]={100,100,200,300,.9f,0};
 NvDsInferLayerInfo layer{};layer.buffer=data;layer.inferDims.numDims=2;
 layer.inferDims.d[0]=1;layer.inferDims.d[1]=6;
 NvDsInferNetworkInfo net{1280,736,3};
 NvDsInferParseDetectionParams params{};params.numClassesConfigured=80;
 params.perClassPreclusterThreshold.assign(80,.25f);
 for(int i=0;i<10000;++i){
  std::vector<NvDsInferParseObjectInfo> objects;
  assert(NvDsInferParseYolo({layer},net,params,objects));
  assert(objects.size()==1);
  const auto& obj=objects[0];
  assert(obj.rotation_angle==0.0f);
  assert(obj.classId==0 && obj.left==100 && obj.top==100 && obj.width==100 && obj.height==200);
  assert(std::abs(obj.detectionConfidence-.9f)<1e-6f);
 }
 std::cout << "10000 parser results: zero rotation, unchanged class/bbox/confidence PASS\n";
}
