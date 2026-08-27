
# CRISP: Cross-Modal Residual Guidance and Spatial Realignment for Remote Sensing Visual Question Answering

# 📢 Abstract

Remote sensing visual question answering (RSVQA) remains challenging because the evidence relevant to the question in remote sensing imagery is typically sparse, spatially dispersed and highly variable in scale and layout. This structural heterogeneity poses a major challenge to existing transfer strategies, which often struggle to achieve effective query-conditioned semantic focusing and localized spatial refinement. To address this issue, we propose CRISP, a task-specific parameter-efficient adaptation framework for RSVQA built on a frozen ViLT backbone. CRISP comprises two complementary components. First, a Cross-Modal Residual Guidance (CMRG) module generates instance-specific guidance tokens from pooled image and question summaries, steering early cross-modal interaction toward queryrelevant content while suppressing background interference. Second, an Attention-Guided Spatial Realignment (ASR) module performs offset-guided feature realignment within intermediate Transformer layers, enabling localized refinement of spatial evidence under scale variation and sparse semantic distribution. Extensive experiments on the RSVQA-LR and RSVQA-HR benchmarks show that CRISP achieves strong overall performance and consistently improves overall accuracy and average accuracy over prior methods, with particularly notable gains on presence, comparison, and region-related questions. These results demonstrate that residual guidance and spatial realignment together provide an effective task-specific parameter-efficient adaptation strategy for RSVQA under the frozen-backbone setting. 



# 🌟 Simple Baseline Model CRISP

<!-- 1. 模型架构图位置 -->
<p align="center">
  <img width="2693" height="1477" alt="frame" src="https://github.com/user-attachments/assets/fdf8eee6-2621-41df-89a7-7db0487cfbeb" />
</p>

---

## 🌈 Results

### Experimental Results on Remote Sensing VQA Benchmarks

#### RSVQA-LR
<!-- 2. 实验结果图表位置（可放图片或 Markdown 表格） -->
<p align="center">
  <img width="1034" height="306" alt="image" src="https://github.com/user-attachments/assets/fc478881-4a24-461b-b147-49f2eead2baa" />
</p>


#### RSVQA-HR Test1
<p align="center">
  <img width="1027" height="301" alt="image" src="https://github.com/user-attachments/assets/d120c357-732d-4a18-810c-49387e1df7de" />
</p>


#### RSVQA-HR Test2
<p align="center">
  <img width="1028" height="302" alt="image" src="https://github.com/user-attachments/assets/8feea79d-f52a-47f5-a011-6ba4457348c6" />
</p>

---

## 🚀 Citation

If you use our data or code in your research or find it helpful, please cite this project:

```bibtex
@article{xu2026crisp,
  title={CRISP: Cross-Modal Residual Guidance and Spatial Realignment for Remote Sensing Visual Question Answering},
  author={Xu, Changhui and Ren, Zhongle and Hou, Biao and Zhang, Cheng and Ning, Jiawei and Li, Weibin and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
