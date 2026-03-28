import Button from "../../ui/Button.jsx";
import Card from "../../ui/Card.jsx";
import { Briefcase, MessageSquare, Lightbulb } from "lucide-react";

const ResultDisplay = ({ data }) => {
  if (!data) return null;

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Role Header */}
      <Card className="p-6 bg-gradient-to-br from-[#edf6f9] to-[#ffddd2]">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-full bg-[#83c5be]/30 text-[#006d77]">
            <Briefcase className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-[#006d77]">Target Role</h2>
            <p className="text-lg text-[#e29578] font-semibold">{data.role}</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Questions Column */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-4">
            <MessageSquare className="w-5 h-5 text-[#e29578]" />
            <h3 className="text-lg font-semibold text-[#006d77]">
              Predicted Interview Questions
            </h3>
          </div>

          <div className="space-y-3">
            {data.interview_questions.map((q, idx) => (
              <Card
                key={idx}
                className="p-4 hover:border-[#e29578] transition-colors"
              >
                <div className="flex gap-3">
                  <span className="flex-shrink-0 flex items-center justify-center w-6 h-6 rounded-full bg-[#e29578] text-xs font-bold text-[#edf6f9]">
                    {idx + 1}
                  </span>
                  <p className="text-sm text-[#006d77]/90 leading-relaxed">
                    {q}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Suggestions Column */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="w-5 h-5 text-[#e29578]" />
            <h3 className="text-lg font-semibold text-[#006d77]">
              Improvement Suggestions
            </h3>
          </div>

          <div className="space-y-3">
            {data.improvement_suggestions.map((s, idx) => (
              <Card
                key={idx}
                className="p-4 border-l-4 border-l-[#e29578] bg-[#ffddd2]/30"
              >
                <p className="text-sm text-[#006d77]/90 leading-relaxed">{s}</p>
              </Card>
            ))}
            {data.improvement_suggestions.length === 0 && (
              <p className="text-sm text-[#83c5be] italic">
                No specific suggestions generated.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;
