#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include <cutlass/util/command_line.h>

class Swizzle
{
public:
    Swizzle(int num_bits, int num_base, int num_shft)
        : m_num_bits(num_bits), m_num_base(num_base), m_num_shft(num_shft)
    {
        assert(m_num_bits >= 0 && "BBits must be positive.");
        assert(m_num_base >= 0 && "MBase must be positive.");
        assert(std::abs(m_num_shft) >= m_num_bits &&
               "abs(SShift) must be more than BBits.");
    }

    template <class Offset>
    auto apply(Offset offset) const noexcept
    {
        return offset ^ shiftr(offset & m_yyy_msk); // ZZZ ^= YYY
    }

    template <class Offset>
    auto operator()(Offset offset) const noexcept
    {
        return apply(offset);
    }

private:
    template <class Offset>
    auto shiftr(Offset offset) const noexcept
    {
        return m_msk_sft >= 0 ? offset >> m_msk_sft : offset << -m_msk_sft;
    }

    int m_num_bits;
    int m_num_base;
    int m_num_shft;

    int m_bit_msk = (1 << m_num_bits) - 1;
    int m_yyy_msk = m_bit_msk << (m_num_base + std::max(0, m_num_shft));
    int m_zzz_msk = m_bit_msk << (m_num_base - std::min(0, m_num_shft));
    int m_msk_sft = m_num_shft;
};

template <class LayoutA, class TikzColorFn = cute::TikzColor_BWx8>
void print_layout_shared_memory_bank_id_latex(
    LayoutA const& layout_a, // (m,n) -> idx
    Swizzle const& swizzle, size_t element_size, std::ofstream& file,
    TikzColorFn color = {}) // lambda(idx) -> tikz color string
{
    CUTE_STATIC_ASSERT_V(cute::rank(layout_a) <= cute::Int<2>{});
    auto layout = cute::append<2>(layout_a, cute::Layout<cute::_1, cute::_0>{});

    // Header
    file << "\\documentclass[convert]{standalone}\n"
         << "\\usepackage{tikz}\n\n"
         << "\\begin{document}\n"
         << "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every "
            "node/.style={minimum size=1cm, outer sep=0pt}]\n\n";
    // Layout
    for (int i = 0; i < cute::size<0>(layout); ++i)
    {
        for (int j = 0; j < cute::size<1>(layout); ++j)
        {
            int idx = layout(i, j);
            int swizzled_idx = swizzle(idx);
            int bank_id = (swizzled_idx * element_size / 4) % 32;
            file << "\\node[fill=" << color(bank_id) << "] at (" << i << ","
                 << j << ") {" << bank_id << "};\n";
        }
    }
    // Grid
    file << "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid ("
         << int(cute::size<0>(layout)) << "," << int(cute::size<1>(layout))
         << ");\n\n";
    // Labels
    for (int i = 0, j = -1; i < cute::size<0>(layout); ++i)
    {
        file << "\\node at (" << i << "," << j << ") {\\Large{\\texttt{" << i
             << "}}};\n";
    }
    for (int i = -1, j = 0; j < cute::size<1>(layout); ++j)
    {
        file << "\\node at (" << i << "," << j << ") {\\Large{\\texttt{" << j
             << "}}};\n";
    }
    // Footer
    file << "\\end{tikzpicture}\n"
         << "\\end{document}\n";
}

std::ostream& print_usage(std::ostream& out)
{
    out << "Print the shared memory bank ids for a given shared memory layout "
           "and swizzle configuration.\n"
        << "\n"
        << "Options:\n"
        << "\n"
        << "  --help                            If specified, displays this "
           "usage statement.\n\n"
        << "  --m=<int>                         Matrix on shared memory M "
           "dimension\n"
        << "  --n=<int>                         Matrix on shared memory N "
           "dimension\n"
        << "  --stride_m=<int>                  Matrix on shared memory M "
           "stride\n"
        << "  --stride_n=<int>                  Matrix on shared memory N "
           "stride\n"
        << "  --element_size=<int>              Element size in bytes\n"
        << "  --swizzle_num_mask_bits=<int>     Number of swizzle mask bits\n"
        << "  --swizzle_num_base=<int>          Number of swizzle base bits\n"
        << "  --swizzle_num_shift=<int>         Number of swizzle shift bits\n"
        << "  --latex_file_path=<string>        LaTeX file path\n";

    out << "\nExamples:\n\n"
        << "$ "
        << " --m=32 --n=64 --stride_m=64 --stride_n=1 --element_size=4 "
           "--swizzle_num_mask_bits=5 --swizzle_num_base=0 "
           "--swizzle_num_shift=6 "
           "--latex_file_path=shared_memory_bank_ids.tex\n";

    return out;
}

int main(int argc, const char** argv)
{

    int m{32};
    int n{64};
    int stride_m{64};
    int stride_n{1};
    int element_size{4};
    int swizzle_num_mask_bits{static_cast<int>(std::log2(32))};
    int swizzle_num_base{0};
    int swizzle_num_shift{static_cast<int>(std::log2(n))};
    std::string latex_file_path{"shared_memory_bank_ids.tex"};

    cutlass::CommandLine cmd(argc, argv);

    if (cmd.check_cmd_line_flag("help"))
    {
        print_usage(std::cout);
        return 0;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("stride_m", stride_m);
    cmd.get_cmd_line_argument("stride_n", stride_n);
    cmd.get_cmd_line_argument("element_size", element_size);
    cmd.get_cmd_line_argument("swizzle_num_mask_bits", swizzle_num_mask_bits);
    cmd.get_cmd_line_argument("swizzle_num_base", swizzle_num_base);
    cmd.get_cmd_line_argument("swizzle_num_shift", swizzle_num_shift);
    cmd.get_cmd_line_argument("latex_file_path", latex_file_path);

    // Print the configurations.
    std::cout << "m: " << m << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "stride_m: " << stride_m << std::endl;
    std::cout << "stride_n: " << stride_n << std::endl;
    std::cout << "element_size: " << element_size << std::endl;
    std::cout << "swizzle_num_mask_bits: " << swizzle_num_mask_bits
              << std::endl;
    std::cout << "swizzle_num_base: " << swizzle_num_base << std::endl;
    std::cout << "swizzle_num_shift: " << swizzle_num_shift << std::endl;
    std::cout << "latex_file_path: " << latex_file_path << std::endl;

    auto shape{cute::make_shape(m, n)};
    auto stride{cute::make_stride(stride_m, stride_n)};
    auto layout{cute::make_layout(shape, stride)};
    auto swizzle{
        Swizzle{swizzle_num_mask_bits, swizzle_num_base, swizzle_num_shift}};

    std::ofstream outfile(latex_file_path);
    if (outfile.is_open())
    {
        print_layout_shared_memory_bank_id_latex(layout, swizzle, element_size,
                                                 outfile);
        outfile.close();
    }
    else
    {
        std::cerr << "Unable to open file." << std::endl;
    }

    return 0;
}
