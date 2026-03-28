import { useState } from "react";
import "./App.css";

import Tabs from "./components/ui/Tabs";
import ResultDisplay from "./components/features/result-display/ResultDisplay";
import FileUpload from "./components/features/resume-form/FileUpload";
import Card from "./components/ui/Card";
import Input from "./components/ui/Input";
import Textarea from "./components/ui/Textarea";
import Button from "./components/ui/Button";
import LoadingSkeleton from "./components/ui/LoadingSkeleton";
import { AlertCircle } from "lucide-react";
import { uploadResume } from "./services/api";

export default function App() {
  const [activeTab, setActiveTab] = useState("input");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [formData, setFormData] = useState({
    file: null,
    jobDescription: "",
    role: "",
  });
  const [formErrors, setFormErrors] = useState({});

  const validateForm = () => {
    const errors = {};
    if (!formData.file) errors.file = "Please upload a PDF resume";
    if (!formData.jobDescription.trim())
      errors.jobDescription = "Job description is required";
    if (!formData.role.trim()) errors.role = "Desired role is required";

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await uploadResume(formData);
      setResult(response);
      setActiveTab("result");
    } catch (err) {
      setError(err.message || "An unexpected error occurred");
      setActiveTab("input");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#edf6f9] text-[#006d77] font-sans selection:bg-[#e29578] selection:text-[#edf6f9]">
      {/* Background decoration */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-b from-[#ffddd2]/60 to-transparent opacity-70"></div>
        <div className="absolute -top-20 -right-20 w-96 h-96 bg-[#83c5be]/20 rounded-full blur-3xl"></div>
        <div className="absolute top-40 -left-20 w-72 h-72 bg-[#e29578]/15 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-12 max-w-5xl">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 text-[#006d77]">
            GenAI Resume Analyzer
          </h1>
          <p className="text-[#83c5be] max-w-2xl mx-auto text-lg">
            Upload your resume and job description to get AI-powered interview
            questions and improvement suggestions tailored to your target role.
          </p>
        </header>

        {/* Main Content */}
        <main>
          <Tabs
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            hasResult={!!result}
          />

          <div className="transition-all duration-300">
            {activeTab === "input" ? (
              <div className="max-w-2xl mx-auto animate-in fade-in zoom-in-95 duration-300">
                <Card className="p-8">
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <FileUpload
                      file={formData.file}
                      onFileChange={(file) =>
                        setFormData((prev) => ({ ...prev, file }))
                      }
                      error={formErrors.file}
                    />

                    <Input
                      id="role"
                      label="Desired Role"
                      placeholder="e.g. Senior Frontend Engineer"
                      value={formData.role}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          role: e.target.value,
                        }))
                      }
                      error={formErrors.role}
                    />

                    <Textarea
                      id="jd"
                      label="Job Description"
                      placeholder="Paste the full job description here..."
                      value={formData.jobDescription}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          jobDescription: e.target.value,
                        }))
                      }
                      error={formErrors.jobDescription}
                    />

                    <div className="pt-4">
                      <Button
                        type="submit"
                        className="w-full"
                        isLoading={loading}
                      >
                        Analyze Resume
                      </Button>
                    </div>
                  </form>
                </Card>
              </div>
            ) : (
              <div>
                {loading ? (
                  <LoadingSkeleton />
                ) : error ? (
                  <div className="text-center py-12">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[#e29578]/20 text-[#e29578] mb-4">
                      <AlertCircle className="w-8 h-8" />
                    </div>
                    <h3 className="text-xl font-semibold text-[#006d77] mb-2">
                      Analysis Failed
                    </h3>
                    <p className="text-[#83c5be] mb-6">{error}</p>
                    <Button
                      variant="secondary"
                      onClick={() => setActiveTab("input")}
                    >
                      Try Again
                    </Button>
                  </div>
                ) : (
                  <ResultDisplay data={result} />
                )}
              </div>
            )}
          </div>
        </main>

        <footer className="mt-20 text-center text-sm text-[#83c5be]">
          <p>&copy; 2026 | SmartPrep — GenAI Resume Analyzer</p>
        </footer>
      </div>
    </div>
  );
}
