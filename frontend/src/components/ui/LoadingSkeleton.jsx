import Card from "./Card.jsx";

const LoadingSkeleton = () => (
  <div className="space-y-8 animate-pulse">
    <Card className="p-6">
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-full bg-[#83c5be]/30"></div>
        <div className="space-y-2">
          <div className="h-6 w-32 bg-[#83c5be]/30 rounded"></div>
          <div className="h-4 w-48 bg-[#83c5be]/30 rounded"></div>
        </div>
      </div>
    </Card>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="space-y-3">
        <div className="h-5 w-48 bg-[#83c5be]/30 rounded mb-4"></div>
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="p-4">
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-[#83c5be]/30"></div>
              <div className="space-y-2 flex-1">
                <div className="h-3 w-full bg-[#83c5be]/30 rounded"></div>
                <div className="h-3 w-5/6 bg-[#83c5be]/30 rounded"></div>
              </div>
            </div>
          </Card>
        ))}
      </div>
      <div className="space-y-3">
        <div className="h-5 w-48 bg-[#83c5be]/30 rounded mb-4"></div>
        {[1, 2].map((i) => (
          <Card key={i} className="p-4">
            <div className="space-y-2">
              <div className="h-3 w-full bg-[#83c5be]/30 rounded"></div>
              <div className="h-3 w-4/5 bg-[#83c5be]/30 rounded"></div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  </div>
);

export default LoadingSkeleton;
