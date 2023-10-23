# SNU-PS
基于分类后验概率空间的孪生Nested-UNet （SNU-PS）变化检测网络                                                                                                                                                          
数据路径                                                                                                                                                                                                                         
--SN7                                                                                                                                                                                                                      
-----Tain                                                                                                                                                                                                                          
---------T1                                                                                                                                                                                                                      
---------T1_Label                                                                                                                                                                                                                
---------T2                                                                                                                                                                                                                    
---------T2_Label                                                                                                                                                                                                  
---------Change_Label                                                                                                                                                                                                      
-----Val                                                                                                                                                                                                                          
---------T1                                                                                                                                                                                                                  
---------T1_Label                                                                                                                                                                                                              
---------T2                                                                                                                                                                                                                    
---------T2_Label                                                                                                                                                                                                                  
---------Change_Label                                                                                                                                                                                                          
-----Test                                                                                                                                                                                                                    
---------T1                                                                                                                                                                                                                  
---------T2                                                                                                                                                                                                                      
CD_Seg：实现语义分割，获取两期影像后验概率。                                                                                                                                                                      
Run.py用于训练，Eval.py用于验证测试。                                                                                                                                                                              
Model和DeeplabV3PLlus_Model包含不同语义分割模型（HRNet/UNet/EfficientNet_Unet/UperNet/UNet_2Plus/UNet_3Plus/DeeplabV3PLlus/TransUNet)                                                                                                
Utils--Model_Select.py 可定义、增加模型、loss、optimizer、scheduler。
      
