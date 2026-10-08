% =========================================================================
% MATLAB Script: Direct Alpha Calculation & CSV Export  (constants updated)
% Based on Updated Derivation: d2I/dV2 = 2 * K0 * alpha
%
% => alpha = d2I/dV2 / (2 * K0)
%
% Justification:
%   Full expression: d2I/dV2 = 2*K0*alpha * [(1 + 2*theta1*Vth) - 3*theta1*alpha*Vds]
%   Peak alpha ~0.01-0.05 (median) =>  3*theta1*alpha*Vds <~ 0.04 (<~4 %)
%   Since theta1*Vth << 1      (~0.5 V^-1 * 0.05 V = 0.025)
%   Therefore: d2I/dV2 = 2*K0*alpha  (V_DS independent, dimensionless alpha)
%   The optional first-order correction (Eq. alpha_first of the SI) can be
%   switched on below; it changes alpha by only ~2-3 %.
%
% Sign convention:
%   +ve d2I/dV2  =>  +ve alpha  =>  conduction enhancement
%   -ve d2I/dV2  =>  -ve alpha  =>  conduction suppression
%
% Units: K0 [A/V^2], d2I/dV2 [A/V^2] => alpha = dimensionless
% NOTE: the Origin derivative columns are per data point (index), so they are
%       divided by dV (1st) and dV^2 (2nd) before use - see section 2.
% Release copy (8 Oct 2026): dV is the mean voltage step (was the median); input path is a placeholder.
% The Python equivalent is alpha_analysis.py / iets/alpha.py.
%
% Constant sources (verify before publication):
%   k_B, eps0, e, h, c ...... CODATA 2018 (Tiesinga et al., RMP 93, 025010, 2021)
%   mu0 (rGO films) ......... Eda, Fanchini & Chhowalla, Nat. Nanotechnol. 3, 270 (2008);
%                             Gomez-Navarro et al., Nano Lett. 7, 3499 (2007)
%   t_ox (GO layer) ......... Schniepp et al., J. Phys. Chem. B 110, 8535 (2006)
%   eps_r (GO layer) ........ GO-dielectric literature: ~3-10 (high frequency)
%   theta1 .................. Sze & Ng (2007); Taur & Ning - thin-oxide MOSFET range
%   Vth ..................... rGO Dirac-point doping ~1e12 cm^-2 (Eda 2008), rescaled to Cox
% =========================================================================
clc; clear; close all;

%% 1. Configuration, Constants & File Paths

%%  Input And Output Directory
device   = 'Impure';      % 'Pure' or 'Impure'  -> picks the matching input folder
base_dir = pwd;           % folder holding '<device> Device/Y1X1_Y2X2.csv' (Origin exports); set as needed
input_dir  = fullfile(base_dir, [device, ' Device']);                 % ...\Pure Device\11_12.csv
output_dir = fullfile(base_dir, 'Alpha_2mm', [device, ' Alpha']);     % new folder, old results untouched

% ---- Fundamental constants (CODATA 2018) --------------------------------
kB    = 1.380649e-23;       % Boltzmann constant          (J/K, exact)
q     = 1.602176634e-19;    % Elementary charge           (C, exact)
eps0  = 8.8541878128e-14;   % Vacuum permittivity         (F/cm)  [= 8.854e-12 F/m]
V2WN  = 8065.544;           % 1 V  ==  8065.544 cm^-1     (e/hc)
T     = 298;                % Measurement temperature     (K)

% ---- Material constants (literature) ------------------------------------
mu0   = 5;                  % Thermally reduced rGO film mobility (cm^2/V*s); lit. range 0.1-10 (films),
                            %   2-200 (single sheets). Upper-mid film value - CITE a thermal-rGO source
eps_r = 10;                 % STATIC (low-frequency) permittivity of the GO layer - appropriate for a
                            %   quasi-DC sweep; high-frequency values are ~3-4, static values >=10 (CITE)
t_ox  = 0.8e-7;             % Functional-layer thickness  (cm) = 0.8 nm (dry GO interlayer ~0.6-0.8 nm);
                            %   lit. range 0.7-1.2 nm

% Oxide (pseudo-dielectric) capacitance  Cox = eps0*eps_r/t_ox
Cox   = eps0 * eps_r / t_ox;   % (F/cm^2)  ~1.11e-5 F/cm^2 = 11.1 uF/cm^2
% NOTE: graphene quantum capacitance (a few uF/cm^2, Xia et al. 2009) acts in
% series; including it would lower the effective Cox by up to ~2x.

% ---- Device geometry (measured) -----------------------------------------
% Electrode grid: rows Y = 1-5 (top to bottom), columns X = 1-4 (left to right).
% File names are Y1X1_Y2X2 (row first, then column), e.g. 23_34 = (2,3)-(3,4).
px  = 0.2;      % Column pitch, centre-to-centre (cm) = 2 mm (measured)
py  = 0.2;      % Row pitch,    centre-to-centre (cm) = 2 mm (measured)
pad = 0;        % Electrode (pad) diameter (cm). 0 => use centre-to-centre distance.
                %   Set it once measured to use the edge-to-edge gap instead.
W   = 0.1;      % Effective channel width (cm) - electrode width in contact with film (CHECK)
A   = 1;        % Cross-sectional area factor (dimensionless)
geom_model = 'strip';   % 'strip'     : K0 ~ W / L               (L = gap per pair)
                        % 'spreading' : K0 ~ pi / ln((d - r)/r)  (2-D contacts, needs pad > 0)

% ---- Second-order model parameters (only used if correction is ON) -----
theta1 = 0.5;   % Mobility-degradation coefficient (V^-1)   range 0.05-1, fit parameter
Vth    = 0.05;  % Effective threshold voltage      (V)      range 0.01-0.1
apply_theta_correction = false;   % true => also write first-order corrected alpha

% K0 = mu0 * Cox * (geometric factor) * A is computed PER ELECTRODE PAIR inside
% the loop, because L differs for horizontal/vertical (2 mm) and diagonal (2.83 mm) pairs.
K0_unit = mu0 * Cox * A;          % (A/V^2) per unit geometric factor

% Ensure output directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Get list of all matching files in the folder
file_list = dir(fullfile(input_dir, '*.csv'));
% keep only pair files named Y1X1_Y2X2.csv (optionally ..._calculation.csv)
keep = ~cellfun(@isempty, regexp({file_list.name}, '^\d\d_\d\d(_calculation)?\.csv$', 'once'));
file_list = file_list(keep);
if isempty(file_list)
    error('No Y1X1_Y2X2.csv files found in:\n  %s\nCheck device / base_dir.', input_dir);
end

fprintf('Cox     = %.4e  F/cm^2  (eps_r = %g, t_ox = %g nm)\n', Cox, eps_r, t_ox*1e7);
fprintf('kB*T    = %.2f meV\n', kB*T/q*1e3);
fprintf('mu0*Cox = %.4e  A/V^2 (times geometric factor per pair)\n', K0_unit);
fprintf('Pitch   = %.2f mm (x), %.2f mm (y); model = %s\n', px*10, py*10, geom_model);
if ~exist('apply_theta_correction', 'var'), apply_theta_correction = false; end
if exist('theta1', 'var') && exist('Vth', 'var')
    fprintf('theta1*Vth = %.3f  (only a check; theta1 is NOT used unless the correction is ON)\n\n', theta1*Vth);
elseif apply_theta_correction
    error('apply_theta_correction = true needs theta1 and Vth to be defined.');
end
fprintf('Found %d file(s) to process.\n\n', length(file_list));

%% 2. Load & Filter CSV Data — Loop Through All Files

for k = 1:length(file_list)

    % Current file details
    current_filename = file_list(k).name;
    input_filename   = fullfile(input_dir, current_filename);

    % Generate output filename  e.g. '11_12.csv' -> '11_12_alpha_new.csv'
    [~, base_name, ~] = fileparts(current_filename);
    prefix            = erase(base_name, '_calculation');
    output_filename   = [prefix, '_alpha_new.csv'];

    %% Pair geometry from the file name  (Y1X1_Y2X2: row first, then column)
    ids = sscanf(strrep(prefix, '_', ' '), '%1d%1d %1d%1d');   % [y1 x1 y2 x2]
    if numel(ids) ~= 4
        warning('Skipping %s: name is not of the form Y1X1_Y2X2', current_filename);
        continue
    end
    dy = abs(ids(3) - ids(1));                 % row difference
    dx = abs(ids(4) - ids(2));                 % column difference
    d  = hypot(dx * px, dy * py);              % centre-to-centre distance (cm)
    switch geom_model
        case 'strip'
            L_pair = d - pad;                  % gap length (cm)
            G_fac  = W / L_pair;
        case 'spreading'
            r      = pad / 2;
            G_fac  = pi / log((d - r) / r);
        otherwise
            error('geom_model must be ''strip'' or ''spreading''');
    end
    if dx == 0 || dy == 0, ptype = 'straight'; else, ptype = 'diagonal'; end
    K0      = K0_unit * G_fac;
    C_total = 1 / (2 * K0);

    %% Load CSV
    original_data  = readtable(input_filename);
    raw_wavenumber = original_data{:, 1};
    raw_V          = original_data{:, 2};
    raw_Y2         = original_data{:, 8};   % Origin 2nd derivative (smoothed)

    % IMPORTANT: Origin differentiated against the ROW INDEX, not against V.
    % (checked: Derivative Y1 equals the current change per data point.)
    % Convert to true derivatives with this file's own voltage step dV:
    %   dI/dV = Y1 / dV ,   d2I/dV2 = Y2 / dV^2
    % Mean step, not the median: in ten impure row-1 files the recorded voltage is quantised to
    % 1 mV (steps 1, 1, 1, 2 mV), so the median is 1.0 mV while the true mean step is 1.25 mV;
    % the median overstated alpha there by (1.25/1.0)^2 = 1.56.
    ok_V    = raw_V(~isnan(raw_V));
    dV      = (ok_V(end) - ok_V(1)) / (numel(ok_V) - 1);   % 1.25e-3 V in every file
    raw_d2I = raw_Y2 / dV^2;                    % (A/V^2)

    % Filter: remove low-voltage noise and negative-wavenumber sweep
    % (wavenumber < 100 cm-1 corresponds to Vds < 100/8065.5 = 0.0124 V — unreliable region)
    valid_mask = raw_wavenumber >= 100;
    wavenumber = raw_wavenumber(valid_mask);
    d2I_dV2    = raw_d2I(valid_mask);

    %% 3. Alpha Calculation — leading order
    alpha_array = C_total * d2I_dV2;
    fprintf('%-6s %-8s d = %.2f mm  dV = %.3f mV  K0 = %.3e A/V^2  ', prefix, ptype, d*10, dV*1e3, K0);

    %% 4. Append Alpha as a Column & Save to Output Directory
    Alpha = NaN(height(original_data), 1);
    Alpha(valid_mask) = alpha_array;
    d2I_dV2_true = NaN(height(original_data), 1);
    d2I_dV2_true(valid_mask) = d2I_dV2;
    Pair_distance_mm = repmat(d * 10, height(original_data), 1);
    K0_used          = repmat(K0,     height(original_data), 1);
    updated_data = [original_data, table(d2I_dV2_true, Alpha, Pair_distance_mm, K0_used)];

    % Optional first-order theta1 correction:
    %   alpha = alpha0 * [1 + theta1*(3*alpha0*Vds - 2*Vth)],  Vds = wavenumber / 8065.544
    if apply_theta_correction
        Vds = wavenumber / V2WN;
        alpha_corr = alpha_array .* (1 + theta1 * (3 * alpha_array .* Vds - 2 * Vth));
        Alpha_corrected = NaN(height(original_data), 1);
        Alpha_corrected(valid_mask) = alpha_corr;
        updated_data = [updated_data, table(Alpha_corrected)];
    end

    full_output_path = fullfile(output_dir, output_filename);
    writetable(updated_data, full_output_path);
    fprintf('-> %s\n', output_filename);

    %% 5. Plot Alpha Spectrum
    figure('Color', 'w', 'Position', [150, 150, 850, 500]);
    plot(wavenumber, alpha_array, 'b-', 'LineWidth', 1.4); hold on;
    if apply_theta_correction
        plot(wavenumber, alpha_corr, 'r--', 'LineWidth', 1.0);
        legend('\alpha (leading order)', '\alpha (\theta_1-corrected)', 'Location', 'best');
    end
    yline(0, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    xlabel('Wavenumber (cm^{-1})', 'FontSize', 11);
    ylabel('\alpha  (dimensionless)', 'FontSize', 11);
    title(sprintf('\\alpha Spectrum — %s (%s, %.2f mm)', strrep(prefix, '_', ' '), ptype, d*10), 'FontSize', 13);
    grid on;

end
