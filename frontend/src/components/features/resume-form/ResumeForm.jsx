import { useState } from "react";
import Button from "../../ui/Button";
import Input from "../../ui/Input";
import Textarea from "../../ui/Textarea";
import FileUpload from "./FileUpload";
// import { FileUpload } from "lucide-react";

const ResumeForm = ({ onSubmit, loading }) => {
  const [file, setFile] = useState(null);
  const [jd, setJd] = useState("");
  const [role, setRole] = useState("");

  const handleSubmit = () => {
    if (!file || !jd || !role) {
      alert("All fields required");
      return;
    }

    const formData = new FormData();
    formData.append("resume", file);
    formData.append("job_description", jd);
    formData.append("role", role);

    onSubmit(formData);
  };

  return (
    <div className="space-y-4">
      <FileUpload file={file} setFile={setFile} />
      <Textarea
        placeholder="Job Description"
        value={jd}
        onChange={(e) => setJd(e.target.value)}
      />
      <Input
        placeholder="Target Role"
        value={role}
        onChange={(e) => setRole(e.target.value)}
      />
      <Button onClick={handleSubmit} disabled={loading}>
        {loading ? "Analyzing..." : "Submit"}
      </Button>
    </div>
  );
};

export default ResumeForm;
