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

import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author Young-Mook Kang, PhD <youngmook@opengsi.org>
 */
public class ClassifierNode {
    private int itsID = -1;
    private int itsLowID = -1;
    private int itsHighID = -1;
    private double itsCriterion = Double.NaN;
    private String itsName = "";
    private int itsIndex = -1;
    private List<Integer> itsNumClasses;
    private double itsDirection = 0;
    private double itsEntropy = 0;
    private double itsRatio = 0;
    
    public ClassifierNode() {
        this.itsID = -1;
        this.itsLowID = -1;
        this.itsHighID = -1;
        this.itsCriterion = Double.NaN;
        this.itsName = "";
        this.itsIndex = -1;
        this.itsDirection = 0;
        this.itsNumClasses = new ArrayList<Integer>();
        this.itsNumClasses.add(0);
        this.itsNumClasses.add(0);
        this.itsEntropy = 0;
        this.itsRatio = 0;
    }
    
    public int next(Instance theInstance) {
        if(theInstance.value(this.getIndex()) > this.getCriterion()) {
            return this.getHighID();
        } 
        return this.getLowID();
    }
    
    public void setNumNClass(int theNum) {
        this.itsNumClasses.set(0, theNum);
    }
    
    public void setNumPClass(int theNum) {
        this.itsNumClasses.set(1, theNum);
    }
    
    public void setNumClasses(int theNum) {
        this.itsNumClasses.clear();
        for(int i=0;i<theNum;i++) {
            this.itsNumClasses.add(0);
        }
    }
    
    public void setEntropy(double theEntropy) {
        this.itsEntropy = theEntropy;
    }
    
    public void setRatio(double theRatio) {
        this.itsRatio = theRatio;
    }
    
    public void setNumClassOf(Object theObject, int theNum) {
        int theIndex = this.itsNumClasses.indexOf(theObject);
        this.itsNumClasses.set(theIndex, theNum);                
    }
    
    public void addOneToClassOfKth(int theIndex) {
        this.itsNumClasses.set(theIndex, this.itsNumClasses.get(theIndex)+1);
    }
    
    public void setID(int theID) {
        this.itsID = theID;
    }
    
    public void setLowID(int theLowID) {
        this.itsLowID = theLowID;        
    }
    
    public void setHighID(int theHighID) {
        this.itsHighID = theHighID;
    }
    public void setCriterion(double theCriterion) {
        this.itsCriterion = theCriterion;
    }
    
    public void setCriterion(String theCriterionString) {
        theCriterionString = theCriterionString.replace("<=", "");
        theCriterionString = theCriterionString.replace(">", "");
        this.itsCriterion = Double.valueOf(theCriterionString);
    }
    
    public void setName(String theName) {
        this.itsName = theName;
    }
    
    public void setIndex(int theIndex) {
        this.itsIndex = theIndex;
    }
    
    public void setDirection(double theDirection) {
        this.itsDirection = theDirection;
    }
    
    public int getID() {
        return this.itsID;
    }
    
    public int getLowID() {
        return this.itsLowID;
    }
    
    public int getHighID() {
        return this.itsHighID;
    }
    
    public String getName() {
        return this.itsName;
    }
    
    public int getIndex() {
        return this.itsIndex;
    }
    
    public double getCriterion() {
        return this.itsCriterion;
    }
    
    public int getNumNClass() {
        return this.itsNumClasses.get(0);
    }
    
    public int getNumPClass() {
        return this.itsNumClasses.get(1);
    }
    
    public double getDirection() {
        return this.itsDirection;
    }
    
    public double getRatio() {
        return this.itsRatio;
    }
    
    public double getEntropy() {
        return this.itsEntropy;
    }
    
    @Override
    public String toString() {
        StringBuilder theBuilder = new StringBuilder();
        theBuilder.append("[");
        theBuilder.append("ID:").append(this.getID()).append(", ");
        theBuilder.append("Name:").append(this.getName()).append(", ");
        theBuilder.append("Criterion:").append(Utils.doubleToString(this.getCriterion(), 6)).append(", ");
        theBuilder.append(this.getLowID()).append(", ");
        theBuilder.append(this.getHighID()).append(", ");
        theBuilder.append("Class:").append(this.itsNumClasses).append(", ");
        theBuilder.append("Entropy:").append(this.getRatio()).append(", ");
        theBuilder.append("Ratio:").append(this.getEntropy()).append(", ");
        theBuilder.append("Direction:").append(this.getDirection()).append(", ");
        theBuilder.append(this.getIndex()).append(" ]");
        
        return theBuilder.toString();
    }
    
    public boolean isLeaf() {
        return (this.itsIndex == -1);
    }
}
