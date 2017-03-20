Bayesian Benchmark Dose (BBMD)
==============================

Bayesian Benchmark Dose (BBMD) is a Python package for conducting benchmark
dose-response modeling for continuous and dichotomous Bayesian models using the
`Stan`_ probabilistic programming language. This repository includes
source code for::

1. statistical implementation of Bayesian benchmark dose analysis of
   dose-response datasets,
2. estimation of benchmark dose values using central tendency and hybrid
   methods, and
3. reporting of results in multiple formats (Microsoft Word, Excel, txt,
   and JSON).

Details on the BBMD modeling system, and performance comparison to other
tools such as the US EPA Benchmark Dose Modeling Software (`BMDS`_) is
documented in a peer-reviewed publication coming soon [submitted; reference
coming soon].

.. _`Stan`: http://mc-stan.org
.. _`BMDS`: https://www.epa.gov/bmds

For more details on BBMD including installation, quickstart, and developer
documentation, see the `documentation`_ section of the github repository. A
companion project which wraps this software with a web-base graphical
user-interface around the software is available in the `bbmd-web`_ repository.

.. _`documentation`: https://github.com/kanshao/bbmd/tree/master/docs
.. _`bbmd-web`: https://github.com/kanshao/bbmd_web

Written by `Kan Shao`_; implemented by `Andy Shapiro`_.

.. _`Kan Shao`: https://info.publichealth.indiana.edu/faculty/current/Shao-Kan.shtml
.. _`Andy Shapiro`: https://github.com/shapiromatron/
