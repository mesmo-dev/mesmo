using Documenter
using FLEDGE

# Build documentation with given settings.
makedocs(
    sitename="FLEDGE",
    pages=[
        "index.md",
        "intro.md",
        "architecture.md",
        "data.md",
        "api.md",
        "contributing.md"
    ]
)

deploydocs(
    repo="github.com/TUMCREATE-ESTL/FLEDGE.jl.git",
    devbranch="develop"
)
