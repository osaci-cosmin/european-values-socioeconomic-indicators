# V1 Survey Weighting Policy
# Politica de ponderare V1

---

## English

### Purpose

The ESS respondent-level data are based on complex probability samples.
Therefore, unweighted sample statistics are not treated as final population
estimates in this project.

The project preserves the original ESS weighting and sample-design variables:

- `dweight`
- `pspwght`
- `pweight`
- `anweight`
- `psu`
- `stratum`

No ESS weight is manually altered.

---

### Weight validation

For the V1 ESS Round 11 analytical sample:

- 49,210 respondents are included.
- 29 countries are included.
- All four ESS weight variables are present for every respondent.
- No weight has a zero or negative value.
- The ESS analysis-weight identity was empirically confirmed:

`anweight = pspwght × pweight`

with a maximum observed numerical difference of:

`0.000000000000`

The population weight (`pweight`) is constant within each country.

Therefore, within a single country, weighted point estimates calculated with
`pspwght` and `anweight` are numerically equivalent.

This equivalence was verified directly for the main analytical variables.

---

### Primary V1 weight

The primary weight for descriptive point estimates is:

`anweight`

This provides one consistent weighting rule across the project.

---

### Country-level estimates

Country-level means and proportions use:

`anweight`

Examples include:

- mean Left-Right self-placement;
- Left / Centre / Right shares;
- mean Social Trust;
- mean centered Human Values scores.

Because `pweight` is constant within each country, these point estimates are
equivalent to estimates calculated using `pspwght`.

---

### Regional and pooled estimates

For analyses combining respondents from multiple countries, the project uses:

`anweight`

Unlike purely within-country estimates, the population-size component matters
when countries are pooled.

This prevents countries with similar ESS sample sizes but very different
population sizes from contributing equally simply because their survey samples
are of comparable size.

---

### Confidence intervals and standard errors

Point-estimate weights alone are not sufficient for survey-design-based
uncertainty estimation.

Confidence intervals and standard errors will additionally account for:

- `psu`
- `stratum`
- `anweight`

The exact variance-estimation procedure is documented separately before
inferential results are produced.

---

### Country-level analytical dataset

After respondent-level weighting, the project creates a country-level dataset
with one observation per country.

Once the unit of analysis becomes the country, respondent weights are not
automatically reused in country-level models.

For example:

`GDP PPS ↔ Mean Social Trust`

is a relationship between countries.

The weighting strategy for such country-level statistical models must therefore
be chosen according to the country-level estimand rather than inherited
mechanically from the respondent-level survey.

---

### Multilevel modelling

Survey weights will not be inserted mechanically into multilevel models.

The treatment and scaling of weights in multilevel analysis will be specified
separately when the multilevel modelling stage is implemented.

---

### Ukraine

In ESS Round 11 Ukraine, `dweight` and `pspwght` are identical in the analytical
data.

The standard ESS analysis-weight relationship remains valid:

`anweight = pspwght × pweight`

Ukraine therefore remains in the primary V1 analytical sample.

---

## Română

### Scop

Datele ESS la nivel de respondent provin din eșantioane probabilistice complexe.
Din acest motiv, statisticile neponderate ale eșantionului nu sunt tratate în
acest proiect drept estimări finale ale populației.

Proiectul păstrează variabilele ESS originale pentru ponderare și designul
eșantionului:

- `dweight`
- `pspwght`
- `pweight`
- `anweight`
- `psu`
- `stratum`

Nicio pondere ESS nu este modificată manual.

---

### Validarea ponderilor

Pentru eșantionul analitic V1 din ESS Round 11:

- sunt incluși 49.210 respondenți;
- sunt incluse 29 de țări;
- toate cele patru variabile de ponderare sunt disponibile pentru fiecare
  respondent;
- nu există ponderi zero sau negative;
- relația ESS pentru analysis weight a fost confirmată empiric:

`anweight = pspwght × pweight`

cu o diferență numerică maximă observată de:

`0.000000000000`

Population weight (`pweight`) este constant în interiorul fiecărei țări.

Prin urmare, în interiorul unei singure țări, estimările punctuale ponderate cu
`pspwght` și `anweight` sunt numeric echivalente.

Această echivalență a fost verificată direct pentru principalele variabile
analitice.

---

### Ponderea principală în V1

Pentru estimările descriptive punctuale folosim:

`anweight`

Astfel păstrăm o regulă unitară de ponderare în cadrul proiectului.

---

### Estimări la nivel de țară

Mediile și proporțiile la nivel de țară folosesc:

`anweight`

Exemple:

- media auto-poziționării Left-Right;
- proporțiile Left / Centre / Right;
- media Social Trust;
- mediile scorurilor centrate Human Values.

Deoarece `pweight` este constant în interiorul unei țări, aceste estimări sunt
echivalente cu cele obținute folosind `pspwght`.

---

### Estimări regionale și pooled

Pentru analize care combină respondenți din mai multe țări folosim:

`anweight`

În aceste analize componenta legată de mărimea populației devine importantă.

Astfel evităm situația în care țări cu populații foarte diferite ar contribui
aproximativ egal doar pentru că dimensiunile eșantioanelor ESS sunt similare.

---

### Confidence intervals și standard errors

Ponderea utilizată pentru point estimates nu este suficientă pentru estimarea
corectă a incertitudinii în cazul unui complex survey design.

Confidence intervals și standard errors vor ține cont suplimentar de:

- `psu`
- `stratum`
- `anweight`

Procedura exactă de estimare a varianței va fi documentată separat înainte de
producerea rezultatelor inferențiale.

---

### Datasetul analitic la nivel de țară

După agregarea respondent-level, proiectul obține un dataset cu o singură
observație pentru fiecare țară.

Din momentul în care unitatea de analiză devine țara, ponderile respondenților
nu sunt reutilizate automat în modelele country-level.

De exemplu:

`GDP PPS ↔ Mean Social Trust`

reprezintă o relație între țări.

Prin urmare, politica de ponderare pentru modelele country-level trebuie să fie
stabilită în funcție de estimandul de la nivelul țării și nu moștenită mecanic
din microdate.

---

### Modele multinivel

Ponderile survey nu vor fi introduse mecanic într-un model multinivel.

Utilizarea și scalarea ponderilor pentru multilevel analysis vor fi stabilite
separat în etapa dedicată modelelor multinivel.

---

### Ucraina

În datele analitice ESS Round 11 pentru Ucraina, `dweight` și `pspwght` sunt
identice.

Relația standard ESS pentru analysis weight rămâne validă:

`anweight = pspwght × pweight`

Prin urmare, Ucraina rămâne în eșantionul analitic principal V1.