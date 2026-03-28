import { Briefcase, CheckCircle } from "lucide-react";

const Tabs = ({ activeTab, setActiveTab, hasResult }) => {
  return (
    <div className="flex space-x-1 rounded-lg bg-[#edf6f9] p-1 border border-[#83c5be] mb-8">
      <button
        onClick={() => setActiveTab('input')}
        className={`flex-1 flex items-center justify-center gap-2 rounded-md py-2.5 text-sm font-medium transition-all ${
          activeTab === 'input' 
            ? 'bg-[#006d77] text-[#edf6f9] shadow-sm' 
            : 'text-[#83c5be] hover:text-[#006d77] hover:bg-[#ffddd2]/50'
        }`}
      >
        <Briefcase className="w-4 h-4" />
        Input Data
      </button>
      <button
        onClick={() => hasResult && setActiveTab('result')}
        disabled={!hasResult}
        className={`flex-1 flex items-center justify-center gap-2 rounded-md py-2.5 text-sm font-medium transition-all ${
          activeTab === 'result' 
            ? 'bg-[#006d77] text-[#edf6f9] shadow-sm' 
            : 'text-[#83c5be] hover:text-[#006d77] hover:bg-[#ffddd2]/50'
        } ${!hasResult ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <CheckCircle className="w-4 h-4" />
        Analysis Result
      </button>
    </div>
  );
};

export default Tabs;