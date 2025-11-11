import React from "react";
import { Typography, Input, Upload, Button } from "antd";
import { InboxOutlined } from "@ant-design/icons";

const { Title, Text } = Typography;
const { Dragger } = Upload;
const { TextArea } = Input;

const ResumeUploadContainer = ({
  props,
  role,
  setRole,
  resumeFile,
  jd,
  onJDInputChange,
  startAnalyze,
  loading,
}) => {
  return (
    <div key="form-container" style={{ padding: "16px 24px" }}>
      <header>
        <Title level={2} style={{ textAlign: "center", marginBottom: "2rem" }}>
          Smart Prep
        </Title>
        <Text
          style={{
            textAlign: "center",
            display: "block",
            fontSize: "16px",
            marginBottom: "2rem",
          }}
        >
          Your personal AI interview coach. Paste your resume, job description,
          and role to get started.
        </Text>
      </header>

      <div
        style={{
          display: "flex",
          flexDirection: "row",
          gap: "24px",
          marginBottom: "1.5rem",
        }}
      >
        {/* Left Column */}
        <section style={{ flex: 1 }}>
          <Title level={5}>1. Upload Your Resume</Title>
          <Dragger {...props}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag your PDF resume to this area
            </p>
          </Dragger>
        </section>

        {/* Right Column */}
      </div>
      <section style={{ flex: 1 }}>
        <Title level={5}>2. Paste the Job Description</Title>
        <TextArea
          value={jd}
          onChange={onJDInputChange}
          rows={10}
          placeholder="Paste the full job description here..."
        />
      </section>

      {/* Bottom Row */}
      <section style={{ marginBottom: "1.5rem" }}>
        <Title level={5}>3. Enter Your Target Role</Title>
        <Input
          value={role}
          onChange={(e) => setRole(e.target.value)}
          placeholder="e.g., Senior Frontend Developer"
          size="large"
        />
      </section>

      <section style={{ textAlign: "center", margin: "2rem 0" }}>
        <Button
          type="primary"
          size="large"
          onClick={startAnalyze}
          loading={loading}
          disabled={!resumeFile || !jd || !role}
        >
          {loading ? "Analyzing..." : "Analyze My Application"}
        </Button>
      </section>
    </div>
  );
};

export default ResumeUploadContainer;
