import React from "react";
import { Typography, Collapse } from "antd";
import { QuestionCircleOutlined, BulbOutlined } from "@ant-design/icons";

const { Title, Text } = Typography;
const { Panel } = Collapse;

const ResumeResponseContainer = ({ output }) => {
  const { interview_questions, improvement_suggestions } = output;

  // Handles the new object format: { experience_based: [], gap_analysis: [] }
  const renderQuestionPanels = (questionsList) => {
    if (!questionsList || questionsList.length === 0) {
      return <Text>No questions generated for this category.</Text>;
    }
    return questionsList.map((question, index) => (
      <Collapse defaultActiveKey={["exp"]} accordion>
        <Panel header={`Question ${index + 1}`} key={index}>
          <p key={index}>{question}</p>
        </Panel>
      </Collapse>
    ));
  };

  return (
    <div
      style={{
        width: "100%",
        display: "flex",
        flexDirection: "row",
        gap: "24px",
        padding: "24px",
      }}
    >
      {/* Left Column: Questions */}
      <div style={{ flex: 2 }}>
        <Title level={4}>
          <QuestionCircleOutlined /> Interview Questions
        </Title>
        {renderQuestionPanels(interview_questions)}
      </div>

      {/* Right Column: Suggestions */}
      <div style={{ flex: 1 }}>
        <Title level={4}>
          <BulbOutlined /> Resume Suggestions
        </Title>
        {!improvement_suggestions || improvement_suggestions.length === 0 ? (
          <Text>Your resume looks well-aligned! No specific suggestions.</Text>
        ) : (
          <Collapse defaultActiveKey={["0"]}>
            {improvement_suggestions?.map((suggestion, index) => (
              <Panel header={`Suggestion ${index + 1}`} key={index}>
                <p>{suggestion}</p>
              </Panel>
            ))}
          </Collapse>
        )}
      </div>
    </div>
  );
};

export default ResumeResponseContainer;
