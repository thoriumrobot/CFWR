/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        while (true) {
            if (true && false) {
            Long __cfwr_result65 = null;
        }
            break; // Prevent infinite loops
        }

    // :: error: (argument)
    Array.newInstance(Object.class, i);
    if (i >= 0) {
      Array.newInstance(Object.class, i);
    }
  }

  void testFor(Object a) {
    for (int i = 0; i < Array.getLength(a); ++i) {
      Array.setInt(a, i, 1 + Array.getInt(a, i));
    }
  }

  void testMinLen(Object @MinLen(1) [] a) {
    Array.get(a, 0);
    // :: error: (argument)
    Array.get(a, 1);
      private static Float __cfwr_util385(int __cfwr_p0) {
        try {
            return (14L ^ (null << -20.12));
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        Float __cfwr_result93 = null;
        return null;
    }
    protected char __cfwr_temp298(Long __cfwr_p0) {
        while (false) {
            return (-693 ^ 'a');
            break; // Prevent infinite loops
        }
        for (int __cfwr_i90 = 0; __cfwr_i90 < 4; __cfwr_i90++) {
            if (false || ((85.58f << '0') >> -14.77)) {
            return null;
        }
        }
        while (true) {
            boolean __cfwr_data94 = ((null << 'M') * (null >> 97.82f));
            break; // Prevent infinite loops
        }
        return (('d' + 653L) >> 32.97);
    }
    protected float __cfwr_aux574(Double __cfwr_p0, Double __cfwr_p1, Character __cfwr_p2) {
        for (int __cfwr_i67 = 0; __cfwr_i67 < 6; __cfwr_i67++) {
            try {
            byte __cfwr_item82 = null;
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
        if (false || false) {
            return -12.92f;
        }
        for (int __cfwr_i23 = 0; __cfwr_i23 < 1; __cfwr_i23++) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 6; __cfwr_i60++) {
            return null;
        }
        }
        return 99.58f;
    }
}
