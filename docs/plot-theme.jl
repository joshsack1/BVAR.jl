# Shared plot styling for the executed @example blocks.
#
# `makedocs(workdir = @__DIR__)` pins the working directory of every evaluated
# block to `docs/`, so each plotting page can pull this in with a bare
# `include("plot-theme.jl")` regardless of how deeply the page is nested.

using PlotThemes
using Plots
using Colors

# Headless rendering: GR otherwise tries to open a display device, which fails
# on a CI runner.
ENV["GKSwstype"] = "100"

const oxocarbon_palette = [
    colorant"#0f62fe", # Blue
    colorant"#525252", # gray
    colorant"#673AB7", # purple
    colorant"#42be65", # Green
    colorant"#FF6F00", # orange
    colorant"#08bdba", # cyan
    colorant"#82cfff", # light blue
]

const oxocarbon_box = PlotTheme(
    Dict([
        :background => colorant"#f2f4f8",
        :framestyle => :box,
        :grid => true,
        :gridalpha => 0.4,
        :linewidth => 1.4,
        :markerstrokewidth => 0,
        :fontfamily => "Computer Modern",
        :colorgradient => :balance,
        :guidefontsize => 12,
        :titlefontsize => 12,
        :tickfontsize => 8,
        :palette => oxocarbon_palette,
        :minorgrid => true,
        :minorticks => 5,
        :gridlinewidth => 0.7,
        :minorgridalpha => 0.06,
        :legend => :outertopright,
    ]),
)

add_theme(:oxocarbon_box, oxocarbon_box)
theme(:oxocarbon_box)
