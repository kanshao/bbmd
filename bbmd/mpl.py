import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt


CBLUE_SHADE = '#B9D2DE'
CBLUE_LINE = '#5F8BC2'
CGRAY = '#808080'
CRED_LINE = '#CF5300'
CGREEN_LINE = '#036564'
CBLUE_POINT = '#2E8B57'

plt.style.use('ggplot')


def model_fit(drs, lines):
    plt.figure(figsize=(11.0, 6.0))

    # show model-fit central line
    plt.plot(
        lines[0], lines[2],
        CRED_LINE, lw=3,
        label='Median'
    )

    # show model-fit 5th to 9th CI plus axis label
    plt.fill_between(
        lines[0], lines[1], lines[3],
        facecolor=CBLUE_SHADE, edgecolor='none', interpolate=True
    )

    # (show axis label since label doesn't work for fill-between)
    plt.plot(
        [], [],
        color=CBLUE_SHADE, linewidth=10,
        label='5th & 95th Percentile'
    )

    # show dose-response points
    plt.plot(
        drs['dose'], drs['ct'], 'o',
        markersize=10,
        markerfacecolor=CBLUE_POINT,
        markeredgecolor=CGRAY,
        label='Input data')
    if drs['errLow'] is not None:
        plt.errorbar(
            drs['dose'], drs['ct'],
            fmt='none',
            yerr=[drs['errLow'], drs['errHigh']],
            ecolor=CGRAY,
            elinewidth=2,
            capsize=7,
            capthick=2,
        )

    plt.xlabel('Dose')
    plt.ylabel('Response')

    extra = (max(lines[0]) - min(lines[0])) * 0.03
    plt.xlim(
        min(lines[0]) - extra,
        max(lines[0]) + extra
    )
    extra = (max(lines[3]) - min(lines[1])) * 0.05
    plt.ylim(
        min(lines[1]) - extra,
        max(lines[3]) + extra,
    )

    # add legend
    plt.legend(loc='best')

    # general plot settings
    plt.margins(0.)
    plt.tick_params(top=False, right=False)
    plt.rcParams.update({'font.size': 14})

    return plt


def parameter_plots(names, params):
    nparams = len(names)
    fig = plt.figure(figsize=(12.0, nparams * 6.0))

    for i, name in enumerate(names):
        vals = params.get(name)

        ax1 = plt.subplot(nparams, 2, i*2+1)
        ax1.hist(vals, bins=14, normed=True, facecolor=CGREEN_LINE)

        # general plot settings
        ax1.tick_params(top=False, right=False)
        ax1.set_title('Parameter %s' % name)

        ax2 = plt.subplot(nparams, 2, i*2+2)
        ax2.plot(range(vals.size), vals, color=CGREEN_LINE)

        # general plot settings
        ax2.tick_params(top=False, right=False)
        ax2.set_title('Parameter %s' % name)
        ax2.set_ylabel('value')
        ax2.set_xlabel('permutation')

        plt.tight_layout()

    return fig


def render_single_kernel(ax, data, color, title):
    ax.plot(
        data['x'], data['y'],
        color, lw=3
    )
    ax.set_title(title)
    plt.tight_layout()


def render_latex(buf, latex):
    mathtext\
        .MathTextParser('bitmap')\
        .to_png(buf, latex, fontsize=12, dpi=300)


def bmd_kernels(isDual, data):
    rows = len(data[0]['models']) + 1
    cols = 2 if isDual else 1
    fig = plt.figure(figsize=(cols * 6., rows * 3.))

    if isDual:
        d1 = data[0]
        d2 = data[1]

        ax = plt.subplot(rows, 2, 1)
        title = 'Model average: {}'.format(d1['dual_type'])
        render_single_kernel(ax, d1['model_average'], CBLUE_LINE, title)

        ax = plt.subplot(rows, 2, 2)
        title = 'Model average: {}'.format(d2['dual_type'])
        render_single_kernel(ax, d2['model_average'], CBLUE_LINE, title)

        for i, m1 in enumerate(d1['models']):
            m2 = d2['models'][i]

            ax = plt.subplot(rows, 2, 2 * (i + 1) + 1)
            title = u'{}: {}'.format(m1['model_name'], d1['dual_type'])
            render_single_kernel(ax, m1['kernel'], CRED_LINE, title)

            ax = plt.subplot(rows, 2, 2 * (i + 1) + 2)
            title = u'{}: {}'.format(m2['model_name'], d2['dual_type'])
            render_single_kernel(ax, m2['kernel'], CRED_LINE, title)

    else:
        d1 = data[0]

        ax = plt.subplot(rows, 1, 1)
        render_single_kernel(ax, d1['model_average'], CBLUE_LINE, 'Model average')

        for i, m1 in enumerate(d1['models']):
            ax = plt.subplot(rows, 1, i + 2)
            render_single_kernel(ax, m1['kernel'], CRED_LINE, m1['model_name'])

    return fig
