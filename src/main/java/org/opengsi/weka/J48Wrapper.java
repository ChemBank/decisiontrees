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
package org.opengsi.weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import javax.sql.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.experiment.Stats;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Young-Mook Kang, PhD <youngmook@opengsi.org>
 */
public class J48Wrapper extends J48 {

    Instances itsData;
    Instances itsAutoScaledData;
    List<Double> itsMeanList;
    List<Double> itsStdDevList;
    List<String> itsClassNameList;
    List<Double> itsRSList;
    List<Double> itsRSWithEntropyList;
    List<Double> itsRSWithRateList;
    List<Double> itsAvgRSList;
    List<Double> itsAvgRSWithEntropyList;
    List<Double> itsAvgRSWithRateList;

    public void openArff(String theArffFile) throws FileNotFoundException, Exception {
        BufferedReader theReader = new BufferedReader(new FileReader(theArffFile));
        this.setData(new Instances(theReader));
        theReader.close();
    }

    public Instances getData() {
        return this.itsData;
    }

    public void setData(Instances theData) throws Exception {
        this.itsData = theData;
        this.itsMeanList = this.getMeanList();
        this.itsStdDevList = this.getStdDevList();
        this.itsRSList = new ArrayList<Double>();
        this.itsRSWithEntropyList = new ArrayList<Double>();
        this.itsRSWithRateList = new ArrayList<Double>();
        this.itsAvgRSList = new ArrayList<Double>();
        this.itsAvgRSWithEntropyList = new ArrayList<Double>();
        this.itsAvgRSWithRateList = new ArrayList<Double>();
        for (int i = 0, size = this.itsData.numInstances(); i < size; i++) {
            this.itsRSList.add(0.0);
            this.itsRSWithEntropyList.add(0.0);
            this.itsRSWithRateList.add(0.0);
            this.itsAvgRSList.add(0.0);
            this.itsAvgRSWithEntropyList.add(0.0);
            this.itsAvgRSWithRateList.add(0.0);
        }
    }

    public List<Double> getRSList() {
        return this.itsRSList;
    }

    public List<Double> getAvgRSList() {
        return this.itsAvgRSList;
    }

    public void autoscale() {
        int theNumInstances = this.getData().numInstances();
        int theNumAttributes = this.getData().numAttributes();
        for (int i = 0; i < theNumInstances; i++) {
            Instance theInstance = this.getData().instance(i);
            for (int ai = 0; ai < theNumAttributes; ai++) {
                if (this.getData().attribute(ai).isNumeric()) {
                    theInstance.setValue(ai, (theInstance.value(ai) - this.itsMeanList.get(ai)) / this.itsStdDevList.get(ai));
                }
            }
        }
        for (int ai = 0; ai < theNumAttributes; ai++) {
            if (this.getData().attribute(ai).isNumeric()) {
                String theNewName = "[(" + this.getData().attribute(ai).name() + "-" + Utils.doubleToString(this.itsMeanList.get(ai), 6) + ")/" + Utils.doubleToString(this.itsStdDevList.get(ai), 6) + "]";
                this.getData().renameAttribute(ai, theNewName);
            }
        }
    }

    public void removeColumn(int theIndex) throws Exception {
        String[] theOptions = new String[2];
        theOptions[0] = "-R";
        theOptions[1] = String.valueOf(theIndex + 1);
        Remove theRemover = new Remove();
        theRemover.setOptions(theOptions);
    }

    public static Instances arffToInstances(String theArffFile) throws FileNotFoundException, IOException {
        BufferedReader theReader = new BufferedReader(new FileReader(theArffFile));
        Instances theData = new Instances(theReader);
        theReader.close();
        return theData;
    }

    public void setClassIndex(int theIndex) {
        this.itsData.setClassIndex(theIndex);
        HashSet<String> theNameSet = new HashSet<String>();
        Attribute theClassAttribute = this.getData().classAttribute();
        int theNumValues = theClassAttribute.numValues();
        for (int i = 0; i < theNumValues; i++) {
            theNameSet.add((theClassAttribute.value(i)));
        }
        this.itsClassNameList = new ArrayList<String>(theNameSet);
    }

    public void buildClassifier() throws Exception {
        this.buildClassifier(this.itsData);
    }

    public void buildCrossValidationModel(int k) throws Exception {
        double thePrecision = 0;
        double theRecall = 0;
        double theFMeasure = 0;
        double theError = 0;

        int theSize = this.itsData.numInstances() / k;
        int theBegin = 0;
        int theEnd = theSize - 1;
        for (int i = 1; i <= k; i++) {
            System.out.println("Iteration: " + i);
            Instances theTrainData = new Instances(this.itsData);
            Instances theTestData = new Instances(this.itsData, theBegin, (theEnd - theBegin));
            for (int j = 0; j < (theEnd - theBegin); j++) {
                theTrainData.delete(theBegin);
            }
            J48 theJ48 = new J48();
            theJ48.buildClassifier(theTrainData);
            Evaluation theEvaluation = new Evaluation(theTestData);
            theEvaluation.evaluateModel(theJ48, theTestData);
            System.out.println("P: " + theEvaluation.precision(1));
            System.out.println("R: " + theEvaluation.recall(1));
            System.out.println("F: " + theEvaluation.fMeasure(1));
            System.out.println("E: " + theEvaluation.errorRate());

            thePrecision += theEvaluation.precision(1);
            theRecall += theEvaluation.recall(1);
            theFMeasure += theEvaluation.fMeasure(1);
            theError += theEvaluation.errorRate();
        }

        Evaluation theEvaluation = new Evaluation(this.itsData);
        theEvaluation.crossValidateModel(this, itsData, k, new Random(1));
        System.out.println("Percent correct: " + Double.toString(theEvaluation.pctCorrect()));
        this.buildClassifier();
        System.out.println(this.graph());
    }

    public List<ClassifierNode> toNodeList() throws Exception {
        List<ClassifierNode> theNodeList = this.m_root.toNodeList();
        int theNumNodes = theNodeList.size();
        for (int i = 0; i < theNumNodes; i++) {
            theNodeList.get(i).setNumClasses(this.itsClassNameList.size());
        }
        int theNumInstances = this.getData().numInstances();
        for (int i = 0; i < theNumInstances; i++) {
            this.__countSamplesInEachNode(this.getData().instance(i), theNodeList);
        }
        for (int i = 0; i < theNodeList.size(); i++) {
            this.__updateDirection(theNodeList, i);
        }
        for (int i = 0; i < theNumInstances; i++) {
            this.itsRSList.set(i, this.__calculateRS(this.getData().instance(i), theNodeList));
            this.itsAvgRSList.set(i, this.__calculateAvgRS(this.getData().instance(i), theNodeList));
        }
        return theNodeList;
    }

    private void __countSamplesInEachNode(Instance theInstance, List<ClassifierNode> theNodeList) {
        int theBegin = 0;
        String theClassName = theInstance.stringValue(theInstance.classIndex());
        do {
            ClassifierNode theNode = theNodeList.get(theBegin);
            theNode.addOneToClassOfKth(this.itsClassNameList.indexOf(theClassName));
            //System.err.println(theInstance.stringValue(theInstance.classIndex()));
            //System.err.println(theNodeList.get(theBegin));
            theBegin = theNodeList.get(theBegin).next(theInstance);
            if (theNodeList.get(theBegin).isLeaf()) {
                ClassifierNode theLeafNode = theNodeList.get(theBegin);
                theLeafNode.addOneToClassOfKth(this.itsClassNameList.indexOf(theClassName));
            }
        } while (!theNodeList.get(theBegin).isLeaf());
    }

    private double __calculateRS(Instance theInstance, List<ClassifierNode> theNodeList) {
        int theBegin = 0;
        double theRS = 0.0;
        while (!theNodeList.get(theBegin).isLeaf()) {
            theRS += this.__calculateRSUnit(theBegin, theInstance, theNodeList);
            theBegin = theNodeList.get(theBegin).next(theInstance);
        }
        return theRS;
    }

    private double __calculateAvgRS(Instance theInstance, List<ClassifierNode> theNodeList) {
        int theBegin = 0;
        double theRS = 0.0;
        double theNumNodes = 0;
        while (!theNodeList.get(theBegin).isLeaf()) {
            theRS += this.__calculateRSUnit(theBegin, theInstance, theNodeList);
            theNumNodes++;
            theBegin = theNodeList.get(theBegin).next(theInstance);
        }
        return theRS / theNumNodes;
    }

    private double __calculateRSUnit(int theIndex, Instance theInstance, List<ClassifierNode> theNodeList) {
        ClassifierNode theNode = theNodeList.get(theIndex);
        double theDistance = theInstance.value(theNode.getIndex()) - theNode.getCriterion();
        double theDirection = theNode.getDirection();
        return theDirection * theDistance;
    }

    private void __updateDirection(List<ClassifierNode> theNodeList, int theIndex) {
        ClassifierNode theNode = theNodeList.get(theIndex);
        if (!theNode.isLeaf()) {
            int lN = theNodeList.get(theNode.getLowID()).getNumNClass();
            int lP = theNodeList.get(theNode.getLowID()).getNumPClass();
            int hN = theNodeList.get(theNode.getHighID()).getNumNClass();
            int hP = theNodeList.get(theNode.getHighID()).getNumPClass();
            theNode.setDirection(this.__calculateDirection(lN, lP, hN, hP));
        }
    }

    private double __calculateDirection(int lN, int lP, int hN, int hP) {
        if (lN > lP) {
            if (hP > hN) {
                return 1.0;
            } else if (lN > hN) {
                return 1.0;
            } else {
                return -1.0;
            }
        } else if (hN > hP) {
            return -1.0;
        } else if (hP > lP) {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    private double __calculateRD(Instance theInstance, ClassifierNode theNode) {

        return 0.0;
    }

    public List<Double> getRelativeStrengthList() throws Exception {
        List<Double> theRSList = new ArrayList<Double>();
        List<ClassifierNode> theNodeList = this.toNodeList();

        this.m_root.toNodeList();
//        double [] theDoubleArray = this.itsData.attributeToDoubleArray(1);
        double[] theDoubleArray = this.itsData.instance(0).toDoubleArray();
        for (int i = 0; i < theDoubleArray.length; i++) {
            theRSList.add(theDoubleArray[i]);
        }

        System.out.println(theRSList);

        return theRSList;

    }

    private double calculateRS(List<ClassifierNode> theNodeList, double[] theValueArray) {
        double theRS = 0.0;
        int theCurrentIndex = 0;
        while (!theNodeList.get(theCurrentIndex).isLeaf()) {
            int theAttributeIndex = theNodeList.get(theCurrentIndex).getIndex();
            //String theAttributeName = theNodeList.get(theCurrentIndex).getName();
            int theHighIndex = theNodeList.get(theCurrentIndex).getHighID();
            int theLowIndex = theNodeList.get(theCurrentIndex).getLowID();
            double theCriterion = theNodeList.get(theCurrentIndex).getCriterion();

            if (theValueArray[theAttributeIndex] > theCriterion) {
                theCurrentIndex = theHighIndex;
            } else {
                theCurrentIndex = theLowIndex;
            }

        }
        return 0;
    }

    private List<Double> doubleArrayToList(double[] theArray) {
        List<Double> theDoubleList = new ArrayList<>();
        for (double d : theArray) {
            theDoubleList.add(d);
        }
        return theDoubleList;
    }

    public List<Double> getMeanList() {
        List<Double> theMeanList = new ArrayList<Double>();
        int theNumAttr = this.getData().numAttributes();
        for (int i = 0; i < theNumAttr; i++) {
            if (this.getData().attribute(i).isNumeric()) {
                Stats theStats = this.getData().attributeStats(i).numericStats;
                theMeanList.add(theStats.mean);
            } else {
                theMeanList.add(Double.NaN);
            }
        }
        return theMeanList;
    }

    public List<Double> getStdDevList() {
        List<Double> theStdDevList = new ArrayList<Double>();
        int theNumAttr = this.getData().numAttributes();
        for (int i = 0; i < theNumAttr; i++) {
            if (this.getData().attribute(i).isNumeric()) {
                Stats theStats = this.getData().attributeStats(i).numericStats;
                theStdDevList.add(theStats.stdDev);
            } else {
                theStdDevList.add(Double.NaN);
            }
        }
        return theStdDevList;
    }
/*
    private void __updateDirection(List<ClassifierNode> theNodeList, int theIndex) {
        ClassifierNode theNode = theNodeList.get(theIndex);
        if (!theNode.isLeaf()) {
            int lN = theNodeList.get(theNode.getLowID()).getNumNClass();
            int lP = theNodeList.get(theNode.getLowID()).getNumPClass();
            int hN = theNodeList.get(theNode.getHighID()).getNumNClass();
            int hP = theNodeList.get(theNode.getHighID()).getNumPClass();
            theNode.setDirection(this.__calculateDirection(lN, lP, hN, hP));
        }
    }
*/
    private void __updateEntropy(List<ClassifierNode> theNodeList, int theIndex) {

    }
    
    private double __calculateEntropy(List<Integer> theValueList) {
        double theEntropy = 0.0;
        int theSize = theValueList.size();
        for(int i=0;i<theSize;i++) {
            if(theValueList.get(i) != 0) {
                theEntropy -= 1.0*theValueList.get(i)*Math.log(theValueList.get(i)/theSize)/theSize;
            }
        }
        return theEntropy / Math.log(2);
    }
    
    private double __calculateRatio(List<Integer> theValueList, int theOrder) {
        double theRatio = 0.0;
        int theSize = theValueList.size();
        double theMax = Collections.max(theValueList).doubleValue();

        return theRatio ;
    }
}
