/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        while (('C' + null)) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 9; __cfwr_i74++) {
            while (true) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 9; __cfwr_i48++) {
            if (true || ((13.79f ^ null) * 'L')) {
            String __cfwr_item95 = "world50";
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }

    v1 = value1.length() -
        try {
            if (false || true) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 6; __cfwr_i75++) {
            if (((-369L * null) & (33.19 | null)) && true) {
            return 252;
        }
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
 3; // condition not satisfied here
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
      public static Double __cfwr_util383(Boolean __cfwr_p0, Long __cfwr_p1, Object __cfwr_p2) {
        if (false && false) {
            while ((38.33 | null)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        if ((null + null) && false) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 5; __cfwr_i69++) {
            while (true) {
            Float __cfwr_elem66 = null;
            break; // Prevent infinite loops
        }
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        return null;
    }
    static Boolean __cfwr_process16(int __cfwr_p0, boolean __cfwr_p1, Float __cfwr_p2) {
        return null;
        if (true || true) {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 2; __cfwr_i95++) {
            try {
            Object __cfwr_entry30 = null;
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        }
        }
        try {
            while ((71.22 << true)) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 7; __cfwr_i17++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        return null;
    }
    private static String __cfwr_compute359() {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        try {
            while (false) {
            while ((52.72 ^ '8')) {
            try {
            if (((true ^ 905L) + -35.74f) && false) {
            try {
            return "world30";
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        return null;
        return "test63";
    }
}
