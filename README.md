# auto-risk-modeling-skill-demo

A vertical Agent Skill demo for credit risk modeling pre-analysis and automated modeling workflows.

This project focuses on the pre-analysis stage of credit risk modeling. It aims to package common modeling preparation tasks, including data preview, column role identification, target variable confirmation, feature statistics, missing value checks, and result export, into a reusable Agent Skill workflow.

## Current Capabilities

- Data file loading and basic data preview
- Column role identification: information columns, target column, and feature columns
- Feature statistics: valid rate, missing rate, descriptive statistics, etc.
- Business missing value detection
- Automatic export of statistical result files
- Workflow validation on million-level sample size

## Roadmap

- Automated modeling workflow
- Classification / regression task identification
- Model training and performance evaluation
- Feature effectiveness validation
- A/B comparison across different data sources or feature sets
- Automated modeling analysis report generation

## Project Positioning

Compared with general-purpose data analysis scripts or standalone modeling notebooks, this project focuses on converting credit risk modeling pre-analysis SOP into an interactive, reusable, and extensible Agent Skill workflow.

## Disclaimer

This repository is a sanitized demo project for showcasing the workflow design and scripting implementation of a credit risk modeling pre-analysis Skill. It does not contain any company data, client data, internal product logic, or real business rules.


# auto-risk-modeling-skill-demo

面向信贷风控建模前置分析与自动化建模场景的 Agent Skill Demo。

本项目聚焦信贷风控建模中的前置分析流程，尝试将样本预览、字段识别、目标变量确认、特征统计、缺失值检测、统计结果导出等高频步骤封装为可复用的 Agent Skill 工作流，用于提升建模前置分析的效率、规范性和复用性。

## 当前能力

- 样本文件读取与基础信息预览
- 字段角色识别：信息列、目标变量列、特征列
- 特征统计分析：有值率、缺失率、描述统计等
- 业务缺失值检测
- 统计结果文件自动导出
- 支持百万级样本规模下的前置分析流程验证

## 后续规划

- 支持自动化建模流程
- 支持分类 / 回归任务识别
- 支持模型训练与效果评估
- 支持特征效果验证
- 支持不同数据源 / 特征集的 A/B 效果对比
- 支持建模分析报告自动生成

## 项目定位

相比通用数据分析脚本或单一建模 Notebook，本项目更关注将信贷风控建模前置分析 SOP 转化为可交互、可复用、可扩展的 Agent Skill 工作流。

## Disclaimer

本仓库为脱敏 Demo 项目，仅用于展示信贷风控建模前置分析 Skill 的流程设计与脚本化实现，不包含任何公司数据、客户数据、内部产品逻辑或真实业务规则。
