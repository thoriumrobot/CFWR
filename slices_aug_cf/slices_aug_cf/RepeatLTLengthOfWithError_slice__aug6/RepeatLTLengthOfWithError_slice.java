/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        while (((true + 'O') - (null / null))) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 9; __cfwr_i18++) {
            return null;
        }
            break; // Prevent infinite loops
        }

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expres
        while (((true | 891) * -4.26)) {
            if (true && (216 >> -434L)) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 7; __cfwr_i77++) {
            while (true) {
            try {
            while (true) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 10; __cfwr_i96++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
sion = "v2", targetValue = "value2", offset = "2", result = true)
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
      protected byte __cfwr_helper372(boolean __cfwr_p0, double __cfwr_p1, int __cfwr_p2) {
        while (true) {
            while (false) {
            if (true && true) {
            for (int __cfwr_i70 = 0; __cfwr_i70 < 3; __cfwr_i70++) {
            while (('d' + null)) {
            while (true) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 6; __cfwr_i45++) {
            double __cfwr_temp90 = -35.54;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (false && (31.41f >> false)) {
            return null;
        }
        while (false) {
            return -78.54f;
            break; // Prevent infinite loops
        }
        return (null << -98.83);
    }
    char __cfwr_calc258() {
        Boolean __cfwr_result28 = null;
        if (true && (471L ^ (15.97f >> -43.01))) {
            while (true) {
            if (true || true) {
            Integer __cfwr_data88 = null;
        }
            break; // Prevent infinite loops
        }
        }
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return '8';
    }
}
