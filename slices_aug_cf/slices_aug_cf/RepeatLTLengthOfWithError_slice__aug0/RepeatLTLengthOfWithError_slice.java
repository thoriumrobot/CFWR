/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        if (true && false) {
            while (false) {
            try {
            try {
            float __cfwr_item53 = -82.10f;
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.lengt
        while (true) {
            while (false) {
            try {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 1; __cfwr_i14++) {
            Boolean __cfwr_temp14 = null;
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
h() - 3;
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
      private Float __cfwr_func914(byte __cfwr_p0, short __cfwr_p1, double __cfwr_p2) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 8; __cfwr_i70++) {
            return 983;
        }
        while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
