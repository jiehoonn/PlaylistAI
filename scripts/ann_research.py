#!/usr/bin/env python3
"""
ANN Library Research & Selection - Phase 2 Scale Preparation

Comprehensive analysis of Approximate Nearest Neighbors libraries
for scaling music recommendation system to FMA Large (100K+ tracks).

Research covers:
- FAISS (Facebook AI Similarity Search)
- hnswlib (Hierarchical Navigable Small World)
- Annoy (Spotify's ANN library)
- Scikit-learn Approximate KNN

Analysis criteria:
- Performance (speed, memory)
- Accuracy vs exact KNN
- Python integration ease
- Scalability characteristics
- Community support & maintenance
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ANNLibraryProfile:
    """Profile of an ANN library with key characteristics."""
    name: str
    maintainer: str
    algorithm: str
    strengths: List[str]
    weaknesses: List[str]
    python_integration: str
    install_command: str
    use_cases: List[str]
    performance_notes: str
    accuracy_vs_exact: str
    memory_efficiency: str
    build_time: str
    query_time: str
    scalability: str
    recommendation_score: int  # 1-10

class ANNResearcher:
    """Research and analysis of ANN libraries for music recommendation."""
    
    def __init__(self):
        self.libraries = self._research_libraries()
        
    def _research_libraries(self) -> Dict[str, ANNLibraryProfile]:
        """Compile research on major ANN libraries."""
        
        return {
            "faiss": ANNLibraryProfile(
                name="FAISS",
                maintainer="Meta (Facebook AI Research)",
                algorithm="Multiple (IVF, HNSW, PQ, etc.)",
                strengths=[
                    "Industry-proven (used by Meta, Instagram)",
                    "Multiple algorithm options (IVF, HNSW, PQ)",
                    "GPU acceleration support",
                    "Excellent for large-scale (millions+ vectors)",
                    "Highly optimized C++ with Python bindings",
                    "Advanced quantization techniques",
                    "Mature and well-documented"
                ],
                weaknesses=[
                    "Complex API with steep learning curve",
                    "Overkill for smaller datasets (<100K)",
                    "Large installation size",
                    "Memory overhead for small datasets",
                    "GPU dependency for best performance"
                ],
                python_integration="Excellent (official Python bindings)",
                install_command="pip install faiss-cpu  # or faiss-gpu",
                use_cases=[
                    "Large-scale production systems (1M+ vectors)",
                    "Multi-modal search (text, image, audio)",
                    "Systems requiring GPU acceleration",
                    "Enterprise applications with complex requirements"
                ],
                performance_notes="Excellent for 1M+ vectors, moderate overhead for <100K",
                accuracy_vs_exact="95-99% depending on configuration",
                memory_efficiency="Moderate (optimized for large scale)",
                build_time="Fast (optimized indexing)",
                query_time="Sub-millisecond for large indices",
                scalability="Excellent (designed for billion-scale)",
                recommendation_score=7  # Good but complex for our scale
            ),
            
            "hnswlib": ANNLibraryProfile(
                name="hnswlib",
                maintainer="Yury Malkov & community",
                algorithm="Hierarchical Navigable Small World (HNSW)",
                strengths=[
                    "Simple, clean Python API",
                    "Excellent performance for medium-scale (10K-1M vectors)",
                    "Low memory overhead",
                    "Fast build times",
                    "High accuracy (often >99% recall)",
                    "Header-only C++ library",
                    "Active community maintenance"
                ],
                weaknesses=[
                    "No GPU support",
                    "Single algorithm (HNSW only)",
                    "Less mature than FAISS",
                    "Limited advanced features",
                    "Memory usage grows with connectivity"
                ],
                python_integration="Excellent (clean Python wrapper)",
                install_command="pip install hnswlib",
                use_cases=[
                    "Medium-scale recommendation systems",
                    "Real-time search applications",
                    "Prototype development",
                    "CPU-only environments"
                ],
                performance_notes="Sweet spot for 100K-1M vectors",
                accuracy_vs_exact="97-99.5% with proper tuning",
                memory_efficiency="Good (efficient graph structure)",
                build_time="Fast (linear-logarithmic)",
                query_time="Sub-millisecond queries",
                scalability="Good (up to few million vectors)",
                recommendation_score=9  # Perfect fit for our use case
            ),
            
            "annoy": ANNLibraryProfile(
                name="Annoy",
                maintainer="Spotify (Erik Bernhardsson)",
                algorithm="Random Projection Trees",
                strengths=[
                    "Spotify-proven for music recommendations",
                    "Memory-mapped indices (very memory efficient)",
                    "Read-only indices (perfect for production)",
                    "Simple API",
                    "Cross-platform compatibility",
                    "Battle-tested in music domain"
                ],
                weaknesses=[
                    "No dynamic updates (rebuild required)",
                    "Lower accuracy than HNSW/FAISS",
                    "Slower build times for large datasets",
                    "Limited to cosine/angular distance",
                    "Less active development recently"
                ],
                python_integration="Good (Python bindings available)",
                install_command="pip install annoy",
                use_cases=[
                    "Music recommendation systems",
                    "Read-only production indices",
                    "Memory-constrained environments",
                    "Spotify-style applications"
                ],
                performance_notes="Good for music, memory-efficient",
                accuracy_vs_exact="90-95% typical",
                memory_efficiency="Excellent (memory-mapped)",
                build_time="Moderate (tree construction)",
                query_time="Fast queries",
                scalability="Good (Spotify uses for millions of tracks)",
                recommendation_score=8  # Great for music, proven track record
            ),
            
            "sklearn_ann": ANNLibraryProfile(
                name="Scikit-learn LSH/NearestNeighbors",
                maintainer="Scikit-learn community",
                algorithm="LSH Forest (deprecated), Ball Tree approximations",
                strengths=[
                    "Already in our dependency stack",
                    "Familiar API",
                    "No additional dependencies",
                    "Good for prototyping"
                ],
                weaknesses=[
                    "LSH Forest deprecated",
                    "Limited ANN options",
                    "Not optimized for large-scale",
                    "Poor performance vs specialized libraries",
                    "Limited accuracy tuning"
                ],
                python_integration="Native (part of scikit-learn)",
                install_command="# Already available in scikit-learn",
                use_cases=[
                    "Quick prototypes",
                    "Educational purposes",
                    "When external dependencies are prohibited"
                ],
                performance_notes="Adequate for small datasets only",
                accuracy_vs_exact="Variable, generally lower",
                memory_efficiency="Poor compared to specialized libraries",
                build_time="Slow",
                query_time="Slow",
                scalability="Poor",
                recommendation_score=3  # Not suitable for production
            )
        }
    
    def analyze_for_fma_large(self) -> Dict[str, Any]:
        """Analyze which library is best for our FMA Large use case."""
        
        # Our requirements based on performance audit
        requirements = {
            "target_vectors": 106574,  # FMA Large track count
            "vector_dimensions": 61,   # MFCC features
            "target_query_time": "<1s",  # Sub-second recommendations
            "accuracy_requirement": ">95%",  # High accuracy needed
            "memory_budget": "<2GB",   # Reasonable memory usage
            "build_time_tolerance": "<10min",  # Reasonable index build
            "deployment": "CPU-only",  # No GPU dependency
            "integration_ease": "High"  # Simple integration preferred
        }
        
        # Score each library for our specific use case
        scores = {}
        for name, lib in self.libraries.items():
            score = self._score_library_for_fma(lib, requirements)
            scores[name] = score
            
        # Rank libraries
        ranked = sorted(scores.items(), key=lambda x: x[1]["total_score"], reverse=True)
        
        return {
            "requirements": requirements,
            "detailed_scores": scores,
            "ranking": ranked,
            "recommendation": self._generate_recommendation(ranked[0])
        }
    
    def _score_library_for_fma(self, lib: ANNLibraryProfile, req: Dict) -> Dict[str, Any]:
        """Score a library against our FMA Large requirements."""
        
        scores = {}
        
        # Performance score (1-10)
        if "100K" in lib.scalability or "million" in lib.scalability:
            scores["performance"] = 9
        elif "large" in lib.scalability.lower():
            scores["performance"] = 7
        else:
            scores["performance"] = 4
            
        # Accuracy score (1-10)
        if "99%" in lib.accuracy_vs_exact:
            scores["accuracy"] = 10
        elif "95%" in lib.accuracy_vs_exact:
            scores["accuracy"] = 8
        elif "90%" in lib.accuracy_vs_exact:
            scores["accuracy"] = 6
        else:
            scores["accuracy"] = 4
            
        # Integration ease (1-10)
        if "excellent" in lib.python_integration.lower():
            scores["integration"] = 10
        elif "good" in lib.python_integration.lower():
            scores["integration"] = 8
        else:
            scores["integration"] = 5
            
        # Memory efficiency (1-10)
        if "excellent" in lib.memory_efficiency.lower():
            scores["memory"] = 10
        elif "good" in lib.memory_efficiency.lower():
            scores["memory"] = 8
        else:
            scores["memory"] = 6
            
        # Deployment simplicity (1-10)
        gpu_penalty = -3 if "GPU" in str(lib.weaknesses) else 0
        complexity_penalty = -2 if "complex" in str(lib.weaknesses).lower() else 0
        scores["deployment"] = 8 + gpu_penalty + complexity_penalty
        scores["deployment"] = max(1, min(10, scores["deployment"]))
        
        # Music domain relevance (1-10)
        music_bonus = 3 if "music" in str(lib.use_cases).lower() or "Spotify" in lib.maintainer else 0
        scores["domain_fit"] = 7 + music_bonus
        scores["domain_fit"] = min(10, scores["domain_fit"])
        
        # Calculate weighted total
        weights = {
            "performance": 0.25,
            "accuracy": 0.25, 
            "integration": 0.15,
            "memory": 0.15,
            "deployment": 0.10,
            "domain_fit": 0.10
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            "individual_scores": scores,
            "total_score": round(total_score, 2),
            "strengths_for_fma": self._identify_fma_strengths(lib),
            "concerns_for_fma": self._identify_fma_concerns(lib)
        }
    
    def _identify_fma_strengths(self, lib: ANNLibraryProfile) -> List[str]:
        """Identify specific strengths for FMA Large use case."""
        strengths = []
        
        if "100K" in lib.performance_notes or "million" in lib.scalability:
            strengths.append("Perfect scale for FMA Large (106K tracks)")
            
        if "99%" in lib.accuracy_vs_exact:
            strengths.append("High accuracy maintains recommendation quality")
            
        if "simple" in lib.python_integration.lower() or "excellent" in lib.python_integration.lower():
            strengths.append("Easy integration with existing codebase")
            
        if "music" in str(lib.use_cases).lower():
            strengths.append("Proven in music recommendation domain")
            
        if "memory" in lib.memory_efficiency.lower() and "good" in lib.memory_efficiency.lower():
            strengths.append("Efficient memory usage for production deployment")
            
        return strengths
    
    def _identify_fma_concerns(self, lib: ANNLibraryProfile) -> List[str]:
        """Identify potential concerns for FMA Large use case."""
        concerns = []
        
        if "complex" in str(lib.weaknesses).lower():
            concerns.append("Complex API may slow development")
            
        if "GPU" in str(lib.weaknesses):
            concerns.append("GPU dependency adds deployment complexity")
            
        if "overkill" in str(lib.weaknesses).lower():
            concerns.append("May be overengineered for our dataset size")
            
        if "90%" in lib.accuracy_vs_exact and "95%" not in lib.accuracy_vs_exact:
            concerns.append("Lower accuracy may impact recommendation quality")
            
        if "deprecated" in str(lib.weaknesses).lower():
            concerns.append("Limited future support and updates")
            
        return concerns
    
    def _generate_recommendation(self, top_choice: tuple) -> Dict[str, Any]:
        """Generate final recommendation based on analysis."""
        name, analysis = top_choice
        lib = self.libraries[name]
        
        return {
            "recommended_library": name,
            "confidence": "High" if analysis["total_score"] > 8 else "Medium",
            "rationale": f"{lib.name} scores {analysis['total_score']}/10 for FMA Large requirements",
            "key_benefits": analysis["strengths_for_fma"][:3],
            "implementation_priority": "High - Critical for FMA Large scalability",
            "next_steps": [
                f"Install {lib.name}: {lib.install_command}",
                "Create prototype integration with existing KNN system",
                "Run accuracy benchmarks vs exact KNN",
                "Performance test with FMA Small as baseline",
                "Develop migration strategy for FMA Large"
            ]
        }
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report."""
        analysis = self.analyze_for_fma_large()
        
        report = []
        report.append("🔍 ANN LIBRARY RESEARCH REPORT")
        report.append("=" * 60)
        report.append(f"Target Scale: {analysis['requirements']['target_vectors']:,} vectors")
        report.append(f"Dimensions: {analysis['requirements']['vector_dimensions']}")
        report.append(f"Performance Target: {analysis['requirements']['target_query_time']}")
        report.append("")
        
        # Library rankings
        report.append("📊 LIBRARY RANKINGS")
        report.append("-" * 30)
        for i, (name, analysis_data) in enumerate(analysis['ranking'], 1):
            lib = self.libraries[name]
            score = analysis_data['total_score']
            report.append(f"{i}. {lib.name}: {score}/10")
            report.append(f"   {lib.algorithm}")
            if i == 1:
                report.append("   ⭐ RECOMMENDED")
            report.append("")
        
        # Detailed recommendation
        rec = analysis['recommendation']
        report.append("🎯 FINAL RECOMMENDATION")
        report.append("-" * 30)
        report.append(f"Library: {rec['recommended_library'].upper()}")
        report.append(f"Confidence: {rec['confidence']}")
        report.append(f"Rationale: {rec['rationale']}")
        report.append("")
        
        report.append("✅ KEY BENEFITS:")
        for benefit in rec['key_benefits']:
            report.append(f"  • {benefit}")
        report.append("")
        
        report.append("🚀 IMPLEMENTATION ROADMAP:")
        for i, step in enumerate(rec['next_steps'], 1):
            report.append(f"  {i}. {step}")
        
        return "\n".join(report)


def main():
    """Run ANN library research and analysis."""
    print("🔍 STARTING ANN LIBRARY RESEARCH")
    print("=" * 50)
    print("Analyzing ANN libraries for FMA Large scalability...")
    print()
    
    researcher = ANNResearcher()
    report = researcher.generate_research_report()
    
    print(report)
    print()
    print("🎯 RESEARCH COMPLETE - Ready for implementation phase!")
    
    return 0


if __name__ == "__main__":
    exit(main())
