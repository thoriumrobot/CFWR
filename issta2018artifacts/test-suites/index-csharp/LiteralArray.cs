// test case for issue #67:
// https://github.com/kelloggm/checker-framework/issues/67

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class LiteralArray {

    public static readonly String[] timeFormat = {
        ("#.#"), ("#.#"), ("#.#"), ("#.#"), ("#.#"),
    };

    public String format() {
        return format(1);
    }

    public String format(int digits) {
        Contract.Requires(digits >= 0 && digits < LiteralArray.timeFormat.Length);
        return "";
    }
}
