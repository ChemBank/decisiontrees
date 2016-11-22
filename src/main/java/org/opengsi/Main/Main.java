/*
 * Copyright (C) 2016 Young-Mook Kang, PhD <youngmook@opengsi.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.opengsi.Main;

import weka.core.Instances;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.opengsi.weka.J48Wrapper;


/**
 *
 * @author Young-Mook Kang, PhD <youngmook@opengsi.org>
 */
public class Main {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        
//        String theAttributeFile = "ICA_HFE.arff";
//        String theAttributeFile = "RV-dataWithClass.arff";
        //String theAttributeFile = "RemovedRV-dataWithClass.arff";
        String theAttributeFile = "HFE_attributes.arff";
        String theHFECompoundNameFile = "HFE_names.arff";
        String theHFEValueFile = "HFE_values.arff";
        
        Instances theNames = J48Wrapper.arffToInstances(theHFECompoundNameFile);
        Instances theHFEValues = J48Wrapper.arffToInstances(theHFEValueFile);
        
        J48Wrapper theJ48 = new J48Wrapper();
        theJ48.openArff(theAttributeFile);
        theJ48.setClassIndex(0);
//        theJ48.setClassIndex(theJ48.getData().numAttributes()-1);
        theJ48.autoscale();
        theJ48.setUnpruned(true);
        theJ48.buildClassifier();
        
        System.err.println(theJ48.toNodeList());
        System.err.println(theJ48);
        
        for(int i = 0; i<theJ48.getRSList().size();i++) {
            System.err.println(theJ48.getRSList().get(i));
        }
        

        
        //theJ48.buildCrossValidationModel(5);
        

        /*
        BufferedReader theReader = new BufferedReader(new FileReader(theAttributeFile));
        String [] theOptions = new String[2];
        theOptions[0] = "-R";
        theOptions[1] = "1-2";
        Remove theRemover = new Remove();
        theRemover.setOptions(theOptions);
        Instances theData = new Instances(theReader);
        theReader.close();
        System.out.print(theData.get(0));
        theRemover.setInputFormat(theData);
        Instances theTrainData = Filter.useFilter(theData, theRemover);
        // setting class attribute
        theTrainData.setClassIndex(0);
        
        J48 theJ48 = new J48Wrapper();
        theJ48.setUnpruned(false);
        theJ48.buildClassifier(theTrainData);
        theJ48.binarySplitsTipText();
        System.out.print(theJ48.graph());
*/
        
    }

}
