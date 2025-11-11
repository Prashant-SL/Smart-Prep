import { useState } from "react";
import axios from "axios";
import {
  Typography,
  message,
  Upload,
  Tabs,
  Spin,
  Alert,
} from "antd";
import ResumeUploadContainer from "./components/ResumeUploadContainer.jsx";
import ResumeResponseContainer from "./components/ResumeResponseContainer.jsx";

const { Title } = Typography;

// const ResumeUploadContainer = ({
//   props,
//   role,
//   setRole,
//   resumeFile,
//   jd,
//   onJDInputChange,
//   startAnalyze,
//   loading,
// }) => {
//   return (
//     <div key="form-container" style={{ padding: "16px 24px" }}>
//       <header>
//         <Title level={2} style={{ textAlign: "center", marginBottom: "2rem" }}>
//           Smart Prep
//         </Title>
//         <Text
//           style={{
//             textAlign: "center",
//             display: "block",
//             fontSize: "16px",
//             marginBottom: "2rem",
//           }}
//         >
//           Your personal AI interview coach. Paste your resume, job description,
//           and role to get started.
//         </Text>
//       </header>

//       <div
//         style={{
//           display: "flex",
//           flexDirection: "row",
//           gap: "24px",
//           marginBottom: "1.5rem",
//         }}
//       >
//         {/* Left Column */}
//         <section style={{ flex: 1 }}>
//           <Title level={5}>1. Upload Your Resume</Title>
//           <Dragger {...props}>
//             <p className="ant-upload-drag-icon">
//               <InboxOutlined />
//             </p>
//             <p className="ant-upload-text">
//               Click or drag your PDF resume to this area
//             </p>
//           </Dragger>
//         </section>

//         {/* Right Column */}
//         <section style={{ flex: 1 }}>
//           <Title level={5}>2. Paste the Job Description</Title>
//           <TextArea
//             value={jd}
//             onChange={onJDInputChange}
//             rows={10}
//             placeholder="Paste the full job description here..."
//           />
//         </section>
//       </div>

//       {/* Bottom Row */}
//       <section style={{ marginBottom: "1.5rem" }}>
//         <Title level={5}>3. Enter Your Target Role</Title>
//         <Input
//           value={role}
//           onChange={(e) => setRole(e.target.value)}
//           placeholder="e.g., Senior Frontend Developer"
//           size="large"
//         />
//       </section>

//       <section style={{ textAlign: "center", margin: "2rem 0" }}>
//         <Button
//           type="primary"
//           size="large"
//           onClick={startAnalyze}
//           loading={loading}
//           disabled={!resumeFile || !jd || !role}
//         >
//           {loading ? "Analyzing..." : "Analyze My Application"}
//         </Button>
//       </section>
//     </div>
//   );
// };

// const ResumeResponseContainer = ({ output }) => {
//   const { interview_questions, improvement_suggestions } = output;

//   // Handles the new object format: { experience_based: [], gap_analysis: [] }
//   const renderQuestionPanels = (questionsList) => {
//     if (!questionsList || questionsList.length === 0) {
//       return <Text>No questions generated for this category.</Text>;
//     }
//     return questionsList.map((question, index) => (
//       <Collapse defaultActiveKey={["exp"]} accordion>
//         <Panel header={`Question ${index + 1}`} key={index}>
//           <p key={index}>{question}</p>
//         </Panel>
//       </Collapse>
//     ));
//   };

//   return (
//     <div
//       style={{
//         width: "100%",
//         display: "flex",
//         flexDirection: "row",
//         gap: "24px",
//         padding: "24px",
//       }}
//     >
//       {/* Left Column: Questions */}
//       <div style={{ flex: 2 }}>
//         <Title level={4}>
//           <QuestionCircleOutlined /> Interview Questions
//         </Title>
//         {/* <Collapse defaultActiveKey={["exp"]} accordion> */}
//         {/* <Panel header="Experience-Based Questions" key="exp"> */}
//         {renderQuestionPanels(interview_questions)}
//         {/* </Panel> */}
//         {/* </Collapse> */}
//       </div>

//       {/* Right Column: Suggestions */}
//       <div style={{ flex: 1 }}>
//         <Title level={4}>
//           <BulbOutlined /> Resume Suggestions
//         </Title>
//         {!improvement_suggestions || improvement_suggestions.length === 0 ? (
//           <Text>Your resume looks well-aligned! No specific suggestions.</Text>
//         ) : (
//           <Collapse defaultActiveKey={["0"]}>
//             {improvement_suggestions?.map((suggestion, index) => (
//               <Panel header={`Suggestion ${index + 1}`} key={index}>
//                 <p>{suggestion}</p>
//               </Panel>
//             ))}
//           </Collapse>
//         )}
//       </div>
//     </div>
//   );
// };

function App() {
  const [loading, setLoading] = useState(false);
  const [resumeFile, setResumeFile] = useState(null);
  const [jd, setJd] = useState("");
  const [role, setRole] = useState("");
  const [activeTab, setActiveTab] = useState("0");
  const [resumeOutput, setResumeOutput] = useState(null);
  const [apiError, setApiError] = useState(null);

  const draggerProps = {
    name: "file",
    multiple: false,
    accept: "application/pdf",
    beforeUpload(file) {
      const isPDF = file.type === "application/pdf";
      if (!isPDF) {
        message.error("Only PDF files are allowed.");
        return Upload.LIST_IGNORE;
      }
      setResumeFile(file);
      return false;
    },
    onRemove() {
      setResumeFile(null);
    },
  };

  const onJDInputChange = (e) => {
    setJd(e.target.value);
  };

  const startAnalyze = async () => {
    if (!resumeFile || !jd || !role) {
      message.warning("Please fill in all three fields.");
      return;
    }

    setLoading(true);
    setApiError(null);
    setResumeOutput(null);

    const formData = new FormData();
    formData.append("file", resumeFile);
    formData.append("job_description", jd);
    formData.append("desired_role", role);

    try {
      const response = await axios.post(
        import.meta.env.VITE_BACKEND_APP_BASE_URL + "/upload-resume",
        formData
      );

      console.log("response.data", response.data);
      setResumeOutput(response.data);
      setActiveTab("1");
    } catch (error) {
      console.error("Error analyzing resume:", error);
      setApiError(
        "Failed to analyze resume. Please check the backend server and try again."
      );
      message.error("An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const tabsGroup = [
    { label: "Prepare Application", key: "0" },
    { label: "Review Your Plan", key: "1" },
  ];

  const tabContent = {
    0: (
      <ResumeUploadContainer
        props={draggerProps}
        role={role}
        setRole={setRole}
        jd={jd}
        onJDInputChange={onJDInputChange}
        startAnalyze={startAnalyze}
        loading={loading}
        resumeFile={resumeFile}
      />
    ),
    1: (
      <div style={{ padding: "24px", minHeight: "400px" }}>
        {loading && (
          <div style={{ textAlign: "center", marginTop: "4rem" }}>
            <Spin size="large" />
            <Title level={5} style={{ marginTop: "1rem" }}>
              Analyzing... This may take a moment.
            </Title>
          </div>
        )}
        {apiError && (
          <Alert message="Error" description={apiError} type="error" showIcon />
        )}
        {resumeOutput && !loading && (
          <ResumeResponseContainer output={resumeOutput} />
        )}
      </div>
    ),
  };

  return (
    <div style={{ width: "80vw", margin: "2rem auto" }}>
      <Tabs
        activeKey={activeTab}
        centered
        onTabClick={(key) => setActiveTab(key)}
        items={tabsGroup.map((tab) => {
          return {
            label: tab.label,
            key: tab.key,
            children: tabContent[tab.key],
            disabled: tab.key === "1" && !resumeOutput && !loading,
          };
        })}
      />
    </div>
  );
}

export default App;
