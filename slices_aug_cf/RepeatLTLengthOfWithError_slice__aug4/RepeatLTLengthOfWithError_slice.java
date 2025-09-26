/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        while (true) {
            return null;
            break; // Prevent infinite loops
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
      protected static double __cfwr_compute289(Boolean __cfwr_p0) {
        while ((false << null)) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 2; __cfwr_i27++) {
            if (false && (null >> (null % 61.13f))) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 7; __cfwr_i73++) {
            while (false) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 5; __cfwr_i92++) {
            while (false) {
            return (680 ^ null);
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        try {
            while ((true >> 171)) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 5; __cfwr_i10++) {
            while (false) {
            while ((true * (true & -73.30))) {
            try {
            try {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 7; __cfwr_i25++) {
            while ((null * (31.87f % -81.23f))) {
            while (true) {
            String __cfwr_node48 = "test32";
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        if (true || true) {
            try {
            return null;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        return -28.49;
    }
}
