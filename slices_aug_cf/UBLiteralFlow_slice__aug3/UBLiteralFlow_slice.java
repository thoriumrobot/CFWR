/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private static @IndexOrLow("#1") int lineStartIndexPartial(
      String s, @GTENegativeOne int lineStart) {
        for (int __cfwr_i63 = 0; __cfwr_i63 < 1; __cfwr_i63++) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 1; __cfwr_i1++) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 3; __cfwr_i14++) {
            String __cfwr_node63 = "data82";
        }
        }
        }

    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf("#1") int lineStartIndexPartial2(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    // :: error: (return)
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf("#1") int lineStartIndex2(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndex3(String s, int start) {
    if (s.length() == 0) {
      // :: error: (return)
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      // :: error: (return)
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      // :: error: (return)
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndex4(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
      Object __cfwr_helper836(Integer __cfwr_p0, short __cfwr_p1) {
        float __cfwr_entry56 = 95.21f;
        for (int __cfwr_i39 = 0; __cfwr_i39 < 6; __cfwr_i39++) {
            try {
            try {
            Boolean __cfwr_elem5 = null;
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    public static byte __cfwr_proc256(Integer __cfwr_p0, byte __cfwr_p1) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 6; __cfwr_i51++) {
            try {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 4; __cfwr_i6++) {
            return -81.94f;
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        return null;
    }
    boolean __cfwr_handle302(Boolean __cfwr_p0, boolean __cfwr_p1) {
        while ((83.68f - null)) {
            try {
            try {
            if (((998 / -88.14f) << 'b') || false) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 2; __cfwr_i59++) {
            return 'F';
        }
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i13 = 0; __cfwr_i13 < 10; __cfwr_i13++) {
            try {
            int __cfwr_var76 = (568L % null);
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        return false;
    }
}
