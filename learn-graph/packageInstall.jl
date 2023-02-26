import Pkg

meta = ["PyCall","YAML","JSON","Logging","Serialization","ProgressMeter","Graphs",
        "ArgParse","CSV","DataFrames","ProgressBars"]


Pkg.update()

for p in meta
    Pkg.add(p)
end
