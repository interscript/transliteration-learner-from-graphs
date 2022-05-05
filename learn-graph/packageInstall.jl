import Pkg

meta = ["PyCall","YAML","JSON","Logging","Serialization","ProgressMeter",
        "CSV","DataFrames","ProgressBars"]


Pkg.update()

for p in meta
    Pkg.add(p)
end
