/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        try {
            if ((null | -60.63f) || (null / (null + 172L))) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 5; __cfwr_i88++) {
            try {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 4; __cfwr_i15++) {
            return -411;
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e27) {
            // ignore
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
      public static double __cfwr_helper403(byte __cfwr_p0, Boolean __cfwr_p1) {
        if (false || (null << -10.88)) {
            for (int __cfwr_i70 = 0; __cfwr_i70 < 7; __cfwr_i70++) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 9; __cfwr_i90++) {
            if ((-16.80f | -72) && true) {
            if (false || (-118L - null)) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 3; __cfwr_i8++) {
            if (false || true) {
            while (false) {
            return "result78";
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }
        }
        }
        for (int __cfwr_i69 = 0; __cfwr_i69 < 8; __cfwr_i69++) {
            double __cfwr_item42 = -10.25;
        }
        return null;
        while ((null & (-54.75 | 919L))) {
            if (true || false) {
            char __cfwr_var54 = 'H';
        }
            break; // Prevent infinite loops
        }
        return (-18.77 >> true);
    }
}
