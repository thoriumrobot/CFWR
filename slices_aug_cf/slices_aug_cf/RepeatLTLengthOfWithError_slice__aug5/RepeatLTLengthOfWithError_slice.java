/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
  @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
  @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
  // :: error: (contracts.postcondition)
  public void withpostconditionsfunc1() {
        return null;

    v1 = value1.length() - 3; // condition not satisfied here
    v2 = value2.length() - 3;
    v3 = value3.length() - 3;
  }

  @EnsuresLTLengthOfIf(expression = "v1", targetValue = "value1", offset = "3", result = true)
  @EnsuresLTLengthOfIf(expression = "v2", targetValue = "value2", offset = "2", result = true)
  @EnsuresLTLengthOfIf(expression = "v3", targetValue = "value3", offset = "1", result = true)
  public boolean withcondp
        if (true || (('O' - null) % null)) {
            if (true && true) {
            while (false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
ostconditionsfunc2() {
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
      public Long __cfwr_proc779(String __cfwr_p0, int __cfwr_p1, float __cfwr_p2) {
        Long __cfwr_item44 = null;
        if (false && true) {
            while ((121 >> 59.04f)) {
            while (false) {
            if (((-8.86f % 36.82f) / null) || (93.00 % (112L - null))) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
    }
    protected byte __cfwr_helper307() {
        int __cfwr_elem84 = 584;
        while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            if (true && false) {
            if (((-3.29 % '1') ^ (91.47 << true)) || false) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 3; __cfwr_i53++) {
            while (false) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 6; __cfwr_i61++) {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 2; __cfwr_i2++) {
            while (true) {
            try {
            for (int __cfwr_i37 = 0; __cfwr_i37 < 1; __cfwr_i37++) {
            if (('5' + (false - 188L)) || true) {
            if ((489L / 316) || ((88.99f * 473) / (191L >> null))) {
            if ((-25.86f >> 51.32) && true) {
            try {
            if (true || false) {
            long __cfwr_elem46 = (593L >> -699L);
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        while ((-59.65 / ('z' << -48L))) {
            Long __cfwr_result87 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
    float __cfwr_calc794() {
        for (int __cfwr_i4 = 0; __cfwr_i4 < 1; __cfwr_i4++) {
            try {
            return null;
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        }
        for (int __cfwr_i56 = 0; __cfwr_i56 < 3; __cfwr_i56++) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 6; __cfwr_i56++) {
            if ((-56.89f + false) && false) {
            if (((728L ^ -79.53) - -844) && false) {
            return 770L;
        }
        }
        }
        }
        return (-149L + 577);
    }
}
