/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        if (true && ('t' << '1')) {
            if (true && (null / '1')) {
            short __cfwr_data76 = null;
        }
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
      short __cfwr_process63(Float __cfwr_p0, long __cfwr_p1, char __cfwr_p2) {
        return (null | (-657 & null));
        return ('P' - null);
    }
    private Boolean __cfwr_process206(Long __cfwr_p0, String __cfwr_p1) {
        while (false) {
            try {
            Integer __cfwr_obj53 = null;
        } catch (Exception __cfwr_e90) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if ((('g' * 'n') * 535L) && false) {
            if ((false & (null | 917L)) || (62.30 | 'l')) {
            while (true) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 7; __cfwr_i85++) {
            byte __cfwr_val38 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        while (false) {
            while (((-37.75f | 18.23f) & null)) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++) {
            Integer __cfwr_node37 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (true || (-604 * (null | 'h'))) {
            String __cfwr_result2 = "result76";
        }
        return null;
    }
}
