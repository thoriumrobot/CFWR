/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewInstance(int i) {
        if ((272L & null) || false) {
            short __cfwr_temp88 = (('s' << 49.25) ^ -57.89);
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
      static long __cfwr_temp997(byte __cfwr_p0, Object __cfwr_p1) {
        long __cfwr_data70 = ((543L % -65.82) >> (-719L >> '8'));
        String __cfwr_val23 = "value20";
        for (int __cfwr_i95 = 0; __cfwr_i95 < 8; __cfwr_i95++) {
            return -468;
        }
        while (((null & 61.65f) * null)) {
            Character __cfwr_item50 = null;
            break; // Prevent infinite loops
        }
        return (617L & 18.30);
    }
    protected short __cfwr_compute6() {
        for (int __cfwr_i71 = 0; __cfwr_i71 < 8; __cfwr_i71++) {
            if (false || false) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 5; __cfwr_i16++) {
            while (false) {
            while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        return null;
    }
}
