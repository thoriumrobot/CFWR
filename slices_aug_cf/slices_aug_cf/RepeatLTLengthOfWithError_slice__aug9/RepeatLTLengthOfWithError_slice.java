/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        try {
            int __cfwr_elem27 = 365;
        } catch (Exception __cfwr_e70) {
            // ignore
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  @EnsuresLTLength
        while (false) {
            try {
            int __cfwr_val44 = 243;
        } catch (Exception __cfwr_e56) {
            // ignore
        }
            break; // Prevent infinite loops
        }
OfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
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
      Integer __cfwr_func539() {
        if ((-1.53f / null) || ((null - true) / null)) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 9; __cfwr_i84++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 10; __cfwr_i97++) {
            return 60.46f;
        }
        }
        }
        if ((20.10f + null) || true) {
            Double __cfwr_item4 = null;
        }
        return null;
    }
    static long __cfwr_handle735(Character __cfwr_p0, Boolean __cfwr_p1, short __cfwr_p2) {
        while (false) {
            try {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 9; __cfwr_i83++) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e74) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        Float __cfwr_result87 = null;
        return 825L;
    }
    int __cfwr_handle443(String __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i16 = 0; __cfwr_i16 < 6; __cfwr_i16++) {
            while (true) {
            return 997L;
            break; // Prevent infinite loops
        }
        }
        return null;
        if (true && true) {
            return null;
        }
        return 347;
    }
}
