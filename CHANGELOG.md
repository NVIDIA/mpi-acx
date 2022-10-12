# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3] 2022-10-12

- Added support for `memOps_v2` that can be enabled at compile time
- Added a compile time option to disable partitioned communication for
  compatibility with MPI libraries that don't support these functions

## [0.2] 2022-08-22

### Changed

- Renamed `MPIX_Init_stream` and `MPIX_Finalize_stream` to `MPIX_Init` and
  `MPIX_Finalize`.

## [0.1] 2022-08-22

- Initial release of MPI-ACX
