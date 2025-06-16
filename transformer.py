from transformers import pipeline

summarizer=pipeline("summarization",model="facebook/bart-large-cnn")

text='''
One of the major advantages of cell culture is the ability to manipulate the physico-chemical (i.e., temperature, pH, osmotic pressure, O2 and CO2 tension) and the physiological environment (i.e., hormone and nutrient concentrations) in which the cells propagate.
While the physiological environment of the culture is not as well defined as its physico-chemical environment, a better understanding of the components of serum, the identification of the growth factors necessary for proliferation, and a better appreciation of the microenvironment of cells in culture (i.e., cell-cell interactions, diffusion of gases, interactions with the matrix) now allow the culture of certain cell lines in serum-free media.
Considerations for creating an optimized cell culture environment for your cells:
pH levels
Most normal mammalian cell lines grow well at pH 7.4, and there is very little variability among different cell strains. However, some transformed cell lines have been shown to grow better at slightly more acidic environments (pH 7.0 – 7.4), and some normal fibroblast cell lines prefer slightly more basic environments (pH 7.4 – 7.7). Insect cell lines such as Sf9 and Sf21 grow optimally at pH 6.2.
CO2 levels
The growth medium controls the pH of the culture and buffers the cells in culture against changes in the pH. Usually, this buffering is achieved by including an organic (e.g., HEPES) or CO2-bicarbonate based buffer. Because the pH of the medium is dependent on the delicate balance of dissolved carbon dioxide (CO2) and bicarbonate (HCO3–), changes in the atmospheric CO2 can alter the pH of the medium. Therefore, it is necessary to use exogenous CO2 when
3
using media buffered with a CO2-bicarbonate based buffer, especially if the cells are cultured in open dishes or transformed cell lines are cultured at high concentrations. While most researchers usually use 5 – 7% CO2 in air, 4 – 10% CO2 is common for most cell culture experiments. However, each medium has a recommended CO2 tension and bicarbonate concentration to achieve the correct pH and osmolality.
Optimal temperatures for various cell lines:
The optimal temperature for cell culture largely depends on the body temperature of the host from which the cells were isolated, and to a lesser degree on the anatomical variation in temperature (e.g., temperature of the skin may be lower than the temperature of skeletal muscle). Overheating is a more serious problem than underheating for cell cultures; therefore, often the temperature in the incubator is set slightly lower than the optimal temperature.
● Most human and mammalian cell lines are maintained at 36°C to 37°C for optimal growth.
● Insect cells are cultured at 27°C for optimal growth; they grow more slowly at lower temperatures and at temperatures between 27°C and 30°C. Above 30°C, the viability of insect cells decreases, and the cells do not recover even after they are returned to 27°C.
● Avian cell lines require 38.5°C for maximum growth. Although these cells can also be maintained at 37°C, they will grow more slowly.
● Cell lines derived from cold-blooded animals (e.g., amphibians, cold-water fish) tolerate a wide temperature range between 15°C and 26°C.
'''


summary=summarizer(text,max_length=150,min_length=40,do_sample=False)

print(summary[0]['summary_text'])