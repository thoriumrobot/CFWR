/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        if (false && true) {
            return null;
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
  public boolean withcondpostconditionsfunc2() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    return true;
  }

  @EnsuresLTLengthOf.List({
    @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"),
    @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  })
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionfunc1() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf.List({
    @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true),
    @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  })
  @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
  public boolean withcondpostconditionfunc2() {
    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    return true;
      private static int __cfwr_aux420(String __cfwr_p0, Boolean __cfwr_p1, String __cfwr_p2) {
        try {
            if ((null * null) && ((-5.72f >> false) ^ 464)) {
            if (true && true) {
            if (false && (-3.62f / (-788 % 'P'))) {
            short __cfwr_entry90 = (525L % null);
        }
        }
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        if ((286 >> ('C' | -389)) && false) {
            try {
            if (((-160 + null) & 'a') || (null & -170L)) {
            try {
            try {
            if (false || ('l' * (-705L | 665L))) {
            return null;
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        }
        if (false && true) {
            return 'F';
        }
        if (((true | null) - (6.68 / -35.75)) && false) {
            short __cfwr_item80 = null;
        }
        return 35;
    }
}
