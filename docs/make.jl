using Documenter
using FLEDGE

# Change the working directory to docs directory when running locally in REPL
# to avoid include errors.
# cd(@__DIR__)

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
